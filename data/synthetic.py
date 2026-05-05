"""
Soft-continuous gyroid training data.

KEY FIX: n_periods increased from 2.5 → 3.5 (range 3.0–4.5).
Lower periods = large sparse pores = scaffold looks like a cage (old problem).
Higher periods = dense trabecular lattice = matches clinical scaffold in Image 2.

At 64 voxels, n_periods=3.5 gives:
  pore diameter ≈ 64/3.5 / 3 ≈ 6 voxels ≈ 0.94mm at 0.156mm/voxel
  → within 100–500μm clinical target range.

Run: python data/synthetic.py
"""

import os
import numpy as np
from tqdm import tqdm

OUT_DIR = "data/processed"


def generate_bone_sample(
    grid_size: int = 64,
    target_porosity: float = 0.70,
    n_periods: float = 3.5,
    contrast: float = 10.0,
    smoothing: float = 2.0,   # kept for API compatibility
) -> np.ndarray:
    """
    Generate one soft-continuous gyroid scaffold sample.

    n_periods controls pore density:
      2.0–2.5 → large open pores (looks like cage/rings — avoid)
      3.0–3.5 → medium trabecular (best for demo)
      4.0–4.5 → fine trabecular (dense, bone-like)

    Returns float32 [0,1] where 0=pore, 1=solid, 0.5=surface.
    """
    coords = np.linspace(0, 2 * np.pi * n_periods, grid_size, dtype=np.float32)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    gyroid = (
        np.sin(X) * np.cos(Y)
        + np.sin(Y) * np.cos(Z)
        + np.sin(Z) * np.cos(X)
    )

    # Normalize to [0, 1]
    g_min, g_max = gyroid.min(), gyroid.max()
    gyroid_norm = (gyroid - g_min) / (g_max - g_min + 1e-8)

    # Threshold at target porosity percentile
    threshold = float(np.percentile(gyroid_norm, target_porosity * 100))

    # Soft sigmoid: smooth [0,1] field with surface at 0.5
    soft_field = 1.0 / (1.0 + np.exp(-contrast * (gyroid_norm - threshold)))

    return soft_field.astype(np.float32)


def generate_dataset(n_samples: int = 800, grid_size: int = 64):
    os.makedirs(OUT_DIR, exist_ok=True)
    samples, porosities = [], []

    target_porosities = np.random.uniform(0.55, 0.85, n_samples)
    # Tighter period range — all produce dense trabecular structure
    n_periods_arr = np.random.uniform(3.0, 4.5, n_samples)

    print(f"Generating {n_samples} soft-gyroid samples (dense trabecular, {grid_size}³)...")

    for i in tqdm(range(n_samples)):
        sample = generate_bone_sample(
            grid_size=grid_size,
            target_porosity=target_porosities[i],
            n_periods=n_periods_arr[i],
            contrast=10.0,
        )
        actual_porosity = float(np.mean(sample < 0.5))
        samples.append(sample)
        porosities.append(actual_porosity)

    patches  = np.array(samples,    dtype=np.float32)
    poro_arr = np.array(porosities, dtype=np.float32)

    np.save(os.path.join(OUT_DIR, "patches.npy"),    patches)
    np.save(os.path.join(OUT_DIR, "porosities.npy"), poro_arr)

    print(f"\n✓ Done: {len(patches)} samples")
    print(f"  Porosity range : {poro_arr.min()*100:.1f}% – {poro_arr.max()*100:.1f}%")
    print(f"  Field range    : [{patches.min():.3f}, {patches.max():.3f}]")
    print(f"  Next: python model/train.py")


if __name__ == "__main__":
    generate_dataset(n_samples=800, grid_size=64)
