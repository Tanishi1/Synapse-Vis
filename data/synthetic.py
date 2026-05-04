"""
Generate synthetic bone-like voxel training data.

Uses smoothed Gaussian noise fields to create sponge-like structures
that mimic trabecular bone micro-architecture.

Run: python data/synthetic.py
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

OUT_DIR = "data/processed"


def generate_bone_sample(grid_size=64, target_porosity=0.7, smoothing=2.0):
    """
    Generate one synthetic bone-like voxel grid.
    Smoothed random noise → threshold → binary sponge structure.
    """
    noise = np.random.randn(grid_size, grid_size, grid_size)
    smoothed = gaussian_filter(noise, sigma=smoothing)
    threshold = np.percentile(smoothed, target_porosity * 100)
    # Below threshold = pore (0), above = solid bone (1)
    voxel_grid = (smoothed >= threshold).astype(np.float32)
    return voxel_grid


def generate_dataset(n_samples=500, grid_size=64):
    """Generate a full dataset of bone-like voxel patches."""
    os.makedirs(OUT_DIR, exist_ok=True)

    samples = []
    porosities = []

    # Vary porosity targets and smoothing to get diverse structures
    target_porosities = np.random.uniform(0.55, 0.85, n_samples)
    smoothings = np.random.uniform(1.5, 3.5, n_samples)

    print(f"Generating {n_samples} synthetic bone samples at {grid_size}³...")

    for i in tqdm(range(n_samples)):
        sample = generate_bone_sample(
            grid_size=grid_size,
            target_porosity=target_porosities[i],
            smoothing=smoothings[i],
        )
        actual_porosity = 1.0 - sample.mean()  # fraction of pore space
        samples.append(sample)
        porosities.append(actual_porosity)

    patches = np.array(samples, dtype=np.float32)    # (N, 64, 64, 64)
    poro_arr = np.array(porosities, dtype=np.float32)  # (N,)

    np.save(os.path.join(OUT_DIR, "patches.npy"), patches)
    np.save(os.path.join(OUT_DIR, "porosities.npy"), poro_arr)

    print(f"\n=== Dataset generated ===")
    print(f"Samples: {len(patches)}")
    print(f"Shape: {patches.shape}")
    print(f"Porosity range: {poro_arr.min()*100:.1f}% — {poro_arr.max()*100:.1f}%")
    print(f"Mean porosity: {poro_arr.mean()*100:.1f}%")
    print(f"Saved to: {OUT_DIR}/")


if __name__ == "__main__":
    generate_dataset(n_samples=500, grid_size=64)
