import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import nibabel as nib


def load_nii(path: str) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    # guard against NaNs/Infs
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(diff * diff))


def find_experiments(root: str) -> List[Tuple[str, str]]:
    """
    Return list of (experiment_dir, recon_path) for subfolders containing recon_vol.nii.gz
    """
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        recon = os.path.join(d, "recon_vol.nii.gz")
        if os.path.isfile(recon):
            out.append((d, recon))
    return out


def main():
    # Allow running this script directly by setting the paths here:
    experiments_root = "/tcmldrive/NogaK/svr_exps/losses_weights_exps"
    original_path = "/tcmldrive/NogaK/noga_experiment_data/scan1/ep2d_diff_64dir_iso1.6_s2p2_new_8.nii"
    vol_idx = 1

    topk = 5

    # You can override these variables above if you want to change the paths or topk

    if not os.path.isdir(experiments_root):
        print(f"Experiments root does not exist: {experiments_root}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(original_path):
        print(f"Original volume not found: {original_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading original volume...", file=sys.stderr)
    try:
        original = load_nii(original_path)[...,1:81, vol_idx]
    # INSERT_YOUR_CODE
        # Standard normalization: (x - mean) / std
        original = (original - np.mean(original)) / (np.std(original) + 1e-8)
    except Exception as e:
        print(f"Failed to load original: {e}", file=sys.stderr)
        sys.exit(1)

    exps = find_experiments(experiments_root)
    if not exps:
        print("No experiments with recon_vol.nii.gz found.")
        sys.exit(0)

    results: List[Tuple[str, float]] = []
    for exp_dir, recon_path in exps:
        try:
            recon = load_nii(recon_path)
            mse = compute_mse(recon, original)
            results.append((exp_dir, mse))
        except Exception as e:
            print(f"Skipping {exp_dir}: {e}", file=sys.stderr)

    if not results:
        print("No valid experiment comparisons.")
        sys.exit(0)

    results.sort(key=lambda x: x[1])

    topk = max(0, min(topk, len(results)))
    print(f"Top {topk} experiments (lowest MSE):")
    for i in range(topk):
        exp_dir, mse = results[i]
        print(f"{i+1}. {exp_dir}  MSE={mse:.6f}")
  

if __name__ == "__main__":
    main()


