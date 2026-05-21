# DORGA

**Decoupled Ordinal Refinement with Geometric Alignment for Chest X-ray Severity Scoring**

> MICCAI 2026 (Early Accept)

## Highlights

- **Average MAE 0.301** on Brixia consensus test set (vs. 0.424 BS-Net, 0.350 PAFE)
- **QWK 0.797** with senior radiologist on external Cohen dataset, matching inter-rater agreement (0.790)
- Addresses **ordinal geometry collapse** -- the compounding interaction of Neural Collapse and Minority Collapse in multi-region ordinal settings

## Pipeline Overview

```
Raw CXR Image (any resolution)
       |
       v
 +-----------------------+
 | 1. Lung Segmentation  |   GUNet: graph-based landmark detection
 |    (1024 x 1024)      |   -> 120 landmarks -> binary lung mask
 +-----------------------+
       |
       v
 +-----------------------+
 | 2. Spatial Alignment   |   STN: mask-conditioned affine transform
 |    (512 x 512)         |   -> spatially normalized CXR
 +-----------------------+
       |
       v
 +-----------------------+
 | 3. DORGA Scoring       |   ViT-B/16 + Ordinal Graph Attention
 |    (512 x 512)         |   -> per-ROI severity grades (0-3)
 +-----------------------+
       |
       v
 6 ROI Severity Scores
```

## Method

### Preprocessing

1. **GUNet Segmentation**: A Graph U-Net predicts 120 anatomical landmarks (44 right-lung + 50 left-lung + 26 heart) from the input CXR. Landmarks are converted to binary lung contour masks via filled polygon rendering.

2. **STN Alignment**: A Spatial Transformer Network takes the lung mask as input and predicts an affine transformation matrix. This matrix is applied to the original CXR to produce a spatially normalized image, correcting for patient positioning and rotation.

### DORGA Model (4 Stages)

**Stage 1: ROI Feature Extraction**
ViT-B/16 (MRM pre-trained) produces patch tokens. Per-patch importance scores are computed via depthwise-pointwise convolution with learnable positional bias. ROI features are obtained by weighted pooling across 6 Brixia regions.

**Stage 2: Ordinal Geometry Construction**
- **Class Anchors**: Equally spaced on a great arc of S^(d-1), constrained by polar loss and ordinal separation loss
- **Severity-Structure Decoupling**: Shared projector produces severity embeddings `u`; region-specific projectors produce structure embeddings `v`, constrained orthogonal to the severity axis `w`
- **Losses**: L_align (orbital drift) + L_1D (polar gradient vanishing) + L_obtuse (wrong-hemisphere correction) + L_orth (structure orthogonality) + L_polar + L_ord

**Stage 3: Inter-ROI Refinement via Graph Attention**
- **Dynamic Prior**: K=7 pattern-specific transition matrices composed via learned pattern distribution
- **Ordinal Message Passing**: Neighbor beliefs propagated through conditional prior, aggregated via structure-aware attention
- **Oracle Attention**: Training-time supervision that teaches attention to weight corrective neighbors (no ground-truth at inference)
- **Confidence-Gated SLERP**: Ordinal variance gates the refinement magnitude on the hypersphere

**Stage 4: Angular Classification**
Independent learnable temperatures for Stage 2 (u) and Stage 3 (h) classifiers. Only Stage 3 logits are used at inference.

## Results

### Brixia Consensus Test Set (N=150, 5-radiologist majority vote)

| Model | ROI1 | ROI2 | ROI3 | ROI4 | ROI5 | ROI6 | Avg MAE |
|-------|------|------|------|------|------|------|---------|
| BS-Net | -- | -- | -- | -- | -- | -- | 0.424 |
| PAFE | 0.30 | 0.38 | 0.31 | 0.32 | 0.37 | 0.43 | 0.350 |
| **DORGA (Ours)** | **0.267** | **0.247** | **0.333** | **0.333** | **0.273** | **0.353** | **0.301** |

### Ablation Study (Consensus Test Set)

| Variant | Avg MAE | Avg CC |
|---------|---------|--------|
| **Full DORGA** | **0.301** | **0.829** |
| w/o decoupling | 0.329 | 0.815 |
| w/o oracle | 0.337 | 0.816 |
| S2-only (no GNN) | 0.347 | 0.817 |

### Cohen External Dataset (N=192, no fine-tuning)

| Comparison | QWK | ICC | MAE |
|------------|-----|-----|-----|
| **Ours vs. Senior** | **0.797** | **0.798** | **0.400** |
| Ours vs. Junior | 0.774 | 0.773 | 0.443 |
| Senior vs. Junior | 0.790 | 0.790 | 0.407 |
| BS-Net vs. Senior | -- | -- | 0.518 |

## Installation

```bash
git clone https://github.com/mai-inha/DORGA.git
cd DORGA
pip install -e .
```

### Requirements
- Python >= 3.9
- PyTorch >= 2.0
- See [requirements.txt](requirements.txt) for full dependencies

### Pre-trained Weights

Download the checkpoint files and place them under `assets/weights/`:

| File | Description | Download |
|------|-------------|----------|
| `seg_weights.pt` | GUNet lung segmentation (shared) | TBD |
| `stn_weights.pth` | STN spatial alignment (shared) | TBD |
| `DORGA_Brixia.pth` | DORGA model -- Brixia (R=6, C=4) | TBD |
| `DORGA_PrivateH.pth` | DORGA model -- PrivateH (R=4, C=5) | TBD |

> Weights will be released upon paper acceptance.

### Supported Configurations

| Config | Dataset | ROI | Classes | ROI Mode |
|--------|---------|-----|---------|----------|
| `configs/default.yaml` | Brixia | 6 ROI | 0-3 (4 grades) | 6-zone |
| `configs/privateh.yaml` | PrivateH | 4 ROI | 0-4 (5 grades) | 4-zone |

## Quick Start

### Step 1: Data Preparation

Prepare your dataset folder with a metadata CSV and raw images:

```
your_dataset/
├── meta.csv
└── original/
    ├── sub0001.png
    ├── sub0002.png
    └── ...
```

**CSV format** -- the metadata CSV must have the following columns:

| Column | Description |
|--------|-------------|
| `Filename` | Image filename (e.g., `sub0001.png`) |
| `split` | One of `train`, `valid`, or `test` |
| label columns | One column per ROI with integer severity grades |

The label column names must match `data.label_cols` in your config.

Example for **Brixia** (6 ROI, grade 0-3):

```csv
Filename,split,brixia1,brixia2,brixia3,brixia4,brixia5,brixia6
sub0001.png,train,0,1,2,0,0,1
sub0002.png,valid,1,1,1,0,0,0
sub0003.png,test,0,0,0,0,0,0
```

Example for **PrivateH** (4 ROI, grade 0-4):

```csv
Filename,split,severity1,severity2,severity3,severity4
sub0001.png,train,0,1,2,0
sub0002.png,valid,1,3,1,0
```

Then update the paths in your config:

```yaml
data:
  meta_csv: "your_dataset/meta.csv"
  original_dir: "your_dataset/original"
  normalized_dir: "your_dataset/Normalize"
  label_cols: ["brixia1", "brixia2", ...]  # match your CSV columns
  train_splits: ["train", "valid"]
  test_split: "test"
```

### Step 2: Preprocessing (GUNet + STN)

Run batch preprocessing to generate aligned images and lung masks:

```bash
python scripts/preprocess.py --config configs/default.yaml
```

This creates the `Normalize/` folder automatically:

```
your_dataset/
├── meta.csv
├── original/                # raw images (input)
└── Normalize/               # auto-generated (output)
    ├── images/              # STN-aligned 512x512 PNG
    │   ├── sub0001.png
    │   └── ...
    └── masks/               # GUNet lung masks 512x512 PNG
        ├── sub0001.png
        └── ...
```

Already-processed images are skipped on re-run.

### Step 3: Training

```bash
python scripts/train.py --config configs/default.yaml
```

Key hyperparameters (see `configs/default.yaml`):
- Backbone: ViT-B/16 with MRM pre-trained weights
- Patterns: K=7 (via K-means on training label vectors)
- Projection dim: 768
- Optimizer: AdamW (per-group lr 1e-5 to 3e-4, cosine decay)
- Epochs: 100, effective batch size: 64

### Step 4: Evaluation

```bash
# Brixia consensus test set
python scripts/test_brixia.py --checkpoint assets/weights/DORGA_Brixia.pth

# Cohen inter-rater agreement analysis
python scripts/test_cohen.py --checkpoint assets/weights/DORGA_Brixia.pth
```

### Inference (single image)

For a single image, preprocessing runs automatically (no Step 2 needed):

```bash
# Brixia (6 ROI, grade 0-3)
python scripts/inference.py \
    --image path/to/cxr.png \
    --config configs/default.yaml

# PrivateH (4 ROI, grade 0-4)
python scripts/inference.py \
    --image path/to/cxr.png \
    --config configs/privateh.yaml
```

## Project Structure

```
DORGA/
├── configs/
│   ├── default.yaml              # Brixia config (R=6, C=4, mask-based ROI)
│   └── privateh.yaml             # PrivateH config (R=4, C=5, quadrant ROI)
├── dorga/                         # Main package
│   ├── preprocessing/             # CXR preprocessing pipeline
│   │   ├── segmentation.py       # GUNet lung segmentation wrapper
│   │   ├── alignment.py          # STN spatial alignment wrapper
│   │   └── pipeline.py           # Combined preprocessing (seg -> align)
│   ├── data/
│   │   ├── dataset.py            # Brixia dataset, ROI mask generation, data loaders
│   │   └── pattern.py            # K-means severity pattern clustering
│   ├── models/
│   │   ├── backbone.py           # ViT-B/16 with MAE/MRM checkpoint loading
│   │   ├── patch_importance.py   # Stage 1: depthwise-pointwise patch scoring + ROI pooling
│   │   ├── projectors.py         # Stage 2: shared (severity) & ROI-specific (structure) projectors
│   │   ├── pattern_prior.py      # Pattern predictor + dynamic Bayesian prior
│   │   ├── graph_attention.py    # Stage 3: decoupled ordinal graph attention + SLERP
│   │   ├── classifier.py         # Angular classifier with decoupled temperatures
│   │   ├── dorga_model.py        # Full model integration (BrixiaViT512Dynamic)
│   │   └── stn.py                # Spatial Transformer Network
│   ├── losses/
│   │   └── losses.py             # CE, projection alignment, oracle attention, KL losses
│   └── utils/
│       ├── training.py           # Optimizer groups, parameter freezing
│       ├── monitor.py            # Metric computation (BSN metrics, GNN stats, etc.)
│       └── visualization.py      # t-SNE, attention heatmaps, loss curves
├── GUNet/                         # Graph U-Net for lung segmentation
│   ├── GUNet_model.py            # Encoder-decoder with graph convolutions
│   ├── GUNet_Utils.py            # Chebyshev graph convolution, pooling
│   └── utils.py                  # Graph matrix generation (lungs + heart anatomy)
├── scripts/
│   ├── preprocess.py             # Batch preprocessing (GUNet + STN -> Normalize/)
│   ├── train.py                  # Training entry point
│   ├── test_brixia.py            # Brixia evaluation with BSN metrics
│   ├── test_cohen.py             # Cohen inter-rater agreement (QWK, ICC, etc.)
│   └── inference.py              # End-to-end single-image inference
├── notebooks/                     # Demo notebooks
├── assets/                        # Figures
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## References

- [1] Signoroni et al., "BS-Net: Learning COVID-19 pneumonia severity on a large CXR dataset," Med. Image Anal. 71, 2021
- [2] Papyan et al., "Prevalence of neural collapse during the terminal phase of deep learning training," PNAS 117(40), 2020
- [3] Fang et al., "Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training," PNAS 118(43), 2021
- [12] Zhou et al., "Advancing radiograph representation learning with masked record modeling," ICLR 2023
- [13] Lee et al., "COVID19 to pneumonia: Multi-region lung severity classification using CNN transformer position-aware feature encoding network," 2024

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
