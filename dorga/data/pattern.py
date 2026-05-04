
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path
import torch


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    label_cols = [f'brixia{i}' for i in range(1, 7)]
    
    print(f"Total samples: {len(df)}")
    print(f"Split distribution:")
    print(df['split'].value_counts())
    
    return df, label_cols


def fit_kmeans_on_trainval(df: pd.DataFrame, label_cols: list, n_clusters: int = 6):
    mask_trainval = df['split'].isin(['train', 'valid'])
    df_trainval = df[mask_trainval]
    
    labels_trainval = df_trainval[label_cols].values.astype(np.float32)
    
    print(f"\nFitting KMeans on train/valid ({len(labels_trainval)} samples)...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_ids_trainval = kmeans.fit_predict(labels_trainval)
    
    cluster_severities = []
    for k in range(n_clusters):
        mask_k = cluster_ids_trainval == k
        mean_severity = labels_trainval[mask_k].sum(axis=1).mean()
        cluster_severities.append((k, mean_severity))
    
    cluster_severities.sort(key=lambda x: x[1])
    cluster_to_pattern = {cluster_id: pattern_id 
                          for pattern_id, (cluster_id, _) in enumerate(cluster_severities)}
    
    print(f"\nCluster to Pattern mapping (sorted by severity):")
    for pattern_id, (cluster_id, severity) in enumerate(cluster_severities):
        n_samples = (cluster_ids_trainval == cluster_id).sum()
        print(f"  Pattern {pattern_id} <- Cluster {cluster_id}: "
              f"severity={severity:.2f}, n={n_samples}")
    
    return kmeans, cluster_to_pattern, cluster_severities


def assign_patterns(df: pd.DataFrame, label_cols: list, kmeans: KMeans,
                    cluster_to_pattern: dict):
    labels_all = df[label_cols].values.astype(np.float32)
    
    cluster_ids_all = kmeans.predict(labels_all)
    
    pattern_ids = np.array([cluster_to_pattern[c] for c in cluster_ids_all])
    
    return pattern_ids


def compute_pattern_priors(df: pd.DataFrame, label_cols: list,
                           n_patterns: int = 6, n_classes: int = 4):
    R = 6
    C = n_classes
    K = n_patterns
    
    mask_trainval = df['split'].isin(['train', 'valid'])
    df_tv = df[mask_trainval]
    
    labels = df_tv[label_cols].values.astype(np.int32)
    pattern_ids = df_tv['pattern'].values
    
    pi_global = np.zeros((R, R, C, C), dtype=np.float32)
    for i in range(R):
        for j in range(R):
            for d in range(C):
                mask_d = labels[:, j] == d
                n_d = mask_d.sum()
                if n_d == 0:
                    pi_global[i, j, :, d] = 1.0 / C
                else:
                    for c in range(C):
                        n_cd = ((labels[:, i] == c) & mask_d).sum()
                        pi_global[i, j, c, d] = n_cd / n_d
    
    pi_patterns = np.zeros((K, R, R, C, C), dtype=np.float32)
    
    for k in range(K):
        mask_k = pattern_ids == k
        labels_k = labels[mask_k]
        N_k = labels_k.shape[0]
        
        if N_k == 0:
            pi_patterns[k] = 1.0 / C
            continue
        
        for i in range(R):
            for j in range(R):
                for d in range(C):
                    mask_d = labels_k[:, j] == d
                    n_d = mask_d.sum()
                    
                    if n_d == 0:
                        pi_patterns[k, i, j, :, d] = 1.0 / C
                    else:
                        for c in range(C):
                            n_cd = ((labels_k[:, i] == c) & mask_d).sum()
                            pi_patterns[k, i, j, c, d] = n_cd / n_d
    
    return pi_global, pi_patterns


def analyze_pattern_distribution(df: pd.DataFrame):
    print("\n" + "="*70)
    print("PATTERN DISTRIBUTION ANALYSIS")
    print("="*70)
    
    for split in ['train', 'valid', 'test']:
        df_split = df[df['split'] == split]
        if len(df_split) == 0:
            continue
        
        print(f"\n{split.upper()} set (n={len(df_split)}):")
        pattern_counts = df_split['pattern'].value_counts().sort_index()
        for p, cnt in pattern_counts.items():
            pct = cnt / len(df_split) * 100
            print(f"  Pattern {p}: {cnt:4d} ({pct:5.1f}%)")


def save_results(df: pd.DataFrame, pi_global: np.ndarray, pi_patterns: np.ndarray,
                 cluster_severities: list, output_dir: Path, csv_path: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_output = output_dir / 'brixia_meta_with_pattern.csv'
    df.to_csv(csv_output, index=False)
    print(f"\nSaved CSV: {csv_output}")
    
    pattern_info = []
    for pattern_id, (cluster_id, severity) in enumerate(cluster_severities):
        mask = df['pattern'] == pattern_id
        mask_trainval = mask & df['split'].isin(['train', 'valid'])
        
        labels_p = df.loc[mask_trainval, [f'brixia{i}' for i in range(1, 7)]].values
        if len(labels_p) > 0:
            from collections import Counter
            mode_labels = [Counter(labels_p[:, r]).most_common(1)[0][0] for r in range(6)]
        else:
            mode_labels = [0] * 6
        
        pattern_info.append({
            'pattern_id': pattern_id,
            'original_cluster': int(cluster_id),
            'mean_severity': float(severity),
            'n_samples_trainval': int(mask_trainval.sum()),
            'n_samples_total': int(mask.sum()),
            'mode_labels': mode_labels,
        })
    
    prior_dict = {
        'pi_global': torch.from_numpy(pi_global),
        'pi_patterns': torch.from_numpy(pi_patterns),
        'n_patterns': len(cluster_severities),
        'pattern_info': pattern_info,
    }
    
    prior_path = output_dir / 'pattern_priors.pt'
    torch.save(prior_dict, prior_path)
    print(f"Saved priors: {prior_path}")
    
    print("\n" + "="*70)
    print("PATTERN INFO SUMMARY")
    print("="*70)
    for info in pattern_info:
        print(f"  Pattern {info['pattern_id']}: "
              f"severity={info['mean_severity']:.2f}, "
              f"n_trainval={info['n_samples_trainval']}, "
              f"mode={info['mode_labels']}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--n_patterns", type=int, default=7)
    args = ap.parse_args()

    CSV_PATH = args.csv
    OUTPUT_DIR = Path(args.output)
    N_PATTERNS = args.n_patterns

    df, label_cols = load_data(CSV_PATH)
    
    kmeans, cluster_to_pattern, cluster_severities = fit_kmeans_on_trainval(
        df, label_cols, n_clusters=N_PATTERNS
    )
    
    pattern_ids = assign_patterns(df, label_cols, kmeans, cluster_to_pattern)
    df['pattern'] = pattern_ids
    
    analyze_pattern_distribution(df)
    
    pi_global, pi_patterns = compute_pattern_priors(
        df, label_cols, n_patterns=N_PATTERNS
    )
    
    save_results(df, pi_global, pi_patterns, cluster_severities, OUTPUT_DIR, CSV_PATH)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - brixia_meta_with_pattern.csv (with pattern column)")
    print(f"  - pattern_priors.pt (Global + Pattern-specific priors)")


if __name__ == "__main__":
    main()
