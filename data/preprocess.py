"""
Preprocessing pipeline: raw Micro-CT volumes → 64³ binary voxel patches.

Handles both:
  - Zenodo format: .npy files that are already 100x100x100 binary arrays
  - Figshare format: TIFF stacks in subfolders (152x152x432 slices)

Output: data/processed/patches.npy  (N, 64, 64, 64) float32
        data/processed/porosities.npy (N,) float32

Run: python data/preprocess.py
"""

import os
import glob
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.filters import threshold_otsu
from skimage import io as skio
from tqdm import tqdm

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
PATCH_SIZE = 64
STRIDE = 32
MIN_BONE_FRAC = 0.10   # skip near-empty patches
MAX_BONE_FRAC = 0.90   # skip near-solid patches


def normalize_volume(vol):
    """Scale intensity values to [0, 1]."""
    vmin, vmax = vol.min(), vol.max()
    if vmax == vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def binarize_volume(vol):
    """
    Convert grayscale CT volume to binary bone/pore mask.
    Uses Otsu's method to find the optimal threshold automatically.
    Result: 1 = bone material, 0 = pore space.
    """
    # Light Gaussian smoothing before thresholding (reduces noise)
    smoothed = gaussian_filter(vol, sigma=0.8)
    thresh = threshold_otsu(smoothed)
    binary = (smoothed > thresh).astype(np.float32)
    return binary


def resample_to_isotropic(vol, target_spacing, current_spacing):
    """
    Resample volume so voxels are cubic (uniform spacing).
    Important when different scan axes have different resolutions.
    """
    scale_factors = [c / target_spacing for c in current_spacing]
    if all(abs(s - 1.0) < 0.05 for s in scale_factors):
        return vol  # already isotropic, skip
    resampled = zoom(vol, scale_factors, order=1)
    return resampled.astype(np.float32)


def extract_patches(binary_vol):
    """
    Sliding window extraction of 64³ patches from a 3D binary volume.
    Filters out patches that are too dense or too empty to be useful.
    """
    patches = []
    D, H, W = binary_vol.shape

    for z in range(0, D - PATCH_SIZE + 1, STRIDE):
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                patch = binary_vol[
                    z : z + PATCH_SIZE,
                    y : y + PATCH_SIZE,
                    x : x + PATCH_SIZE,
                ]
                bone_frac = patch.mean()
                if MIN_BONE_FRAC <= bone_frac <= MAX_BONE_FRAC:
                    patches.append(patch)

    return patches


def load_npy_volume(filepath):
    """Load Zenodo-format .npy bone volumes (already 3D arrays)."""
    vol = np.load(filepath).astype(np.float32)
    print(f"    Shape: {vol.shape}, dtype: {vol.dtype}")

    # Check if already binary (values only 0 and 1)
    unique_vals = np.unique(vol)
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
        print("    Already binary — skipping thresholding.")
        return vol
    else:
        # Grayscale volume — normalize then threshold
        vol = normalize_volume(vol)
        return binarize_volume(vol)


def load_tiff_stack(folder_path):
    """
    Load a TIFF stack from a folder (Figshare format).
    Each .tif file is one 2D slice. Stack them into a 3D volume.
    """
    tiff_files = sorted(
        glob.glob(os.path.join(folder_path, "*.tif"))
        + glob.glob(os.path.join(folder_path, "*.tiff"))
        + glob.glob(os.path.join(folder_path, "*.TIF"))
    )
    if not tiff_files:
        return None

    slices = []
    for f in tiff_files:
        img = skio.imread(f)
        if img.ndim == 3:
            img = img[:, :, 0]  # take first channel if RGB
        slices.append(img.astype(np.float32))

    vol = np.stack(slices, axis=0)  # shape: (D, H, W)
    vol = normalize_volume(vol)
    return binarize_volume(vol)


def load_single_tiff(filepath):
    """Load a single multi-page TIFF file as a 3D volume."""
    vol = skio.imread(filepath)
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]  # treat as single slice
    vol = vol.astype(np.float32)
    vol = normalize_volume(vol)
    return binarize_volume(vol)


def pad_small_volume(vol, target=64):
    """
    If volume is smaller than 64 in any dimension,
    pad with zeros (pore space) to reach target size.
    """
    D, H, W = vol.shape
    pd = max(0, target - D)
    ph = max(0, target - H)
    pw = max(0, target - W)
    if pd + ph + pw == 0:
        return vol
    return np.pad(vol, ((0, pd), (0, ph), (0, pw)), mode="constant", constant_values=0)


def process_all():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_patches = []
    all_porosities = []

    raw_files = (
        glob.glob(os.path.join(RAW_DIR, "*.npy"))
        + glob.glob(os.path.join(RAW_DIR, "*.tif"))
        + glob.glob(os.path.join(RAW_DIR, "*.tiff"))
        + glob.glob(os.path.join(RAW_DIR, "*.TIF"))
    )
    raw_dirs = [
        d for d in glob.glob(os.path.join(RAW_DIR, "*")) if os.path.isdir(d)
    ]

    if not raw_files and not raw_dirs:
        print("No raw data found in data/raw/")
        print("Run: python data/download_data.py first.")
        return

    print(f"Found {len(raw_files)} files and {len(raw_dirs)} folders in data/raw/")

    # Process .npy files (Zenodo format)
    for fpath in tqdm(raw_files, desc="Processing files"):
        ext = os.path.splitext(fpath)[1].lower()
        print(f"\n  Processing: {os.path.basename(fpath)}")
        try:
            if ext == ".npy":
                vol = load_npy_volume(fpath)
            else:
                vol = load_single_tiff(fpath)

            # If volume is smaller than 64, pad it
            if min(vol.shape) < PATCH_SIZE:
                vol = pad_small_volume(vol, PATCH_SIZE)

            # If volume is exactly 64³ or smaller, use as-is
            if all(d == PATCH_SIZE for d in vol.shape):
                bone_frac = vol.mean()
                if MIN_BONE_FRAC <= bone_frac <= MAX_BONE_FRAC:
                    all_patches.append(vol)
                    all_porosities.append(1.0 - bone_frac)
                    print(f"    Porosity: {(1 - bone_frac)*100:.1f}%")
            else:
                patches = extract_patches(vol)
                print(f"    Extracted {len(patches)} patches")
                for p in patches:
                    porosity = 1.0 - p.mean()
                    all_patches.append(p)
                    all_porosities.append(porosity)

        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    # Process TIFF stack folders (Figshare format)
    for folder in tqdm(raw_dirs, desc="Processing folders"):
        print(f"\n  Processing folder: {os.path.basename(folder)}")
        try:
            vol = load_tiff_stack(folder)
            if vol is None:
                print("    No TIFF files found, skipping.")
                continue
            patches = extract_patches(vol)
            print(f"    Extracted {len(patches)} patches")
            for p in patches:
                porosity = 1.0 - p.mean()
                all_patches.append(p)
                all_porosities.append(porosity)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue

    if not all_patches:
        print("\nNo patches extracted. Check your data/raw/ folder.")
        return

    patches_arr = np.array(all_patches, dtype=np.float32)   # (N, 64, 64, 64)
    porosities_arr = np.array(all_porosities, dtype=np.float32)  # (N,)

    np.save(os.path.join(OUT_DIR, "patches.npy"), patches_arr)
    np.save(os.path.join(OUT_DIR, "porosities.npy"), porosities_arr)

    print(f"\n=== Preprocessing complete ===")
    print(f"Total patches: {len(patches_arr)}")
    print(f"Porosity range: {porosities_arr.min()*100:.1f}% — {porosities_arr.max()*100:.1f}%")
    print(f"Mean porosity:  {porosities_arr.mean()*100:.1f}%")
    print(f"Saved to: {OUT_DIR}/patches.npy")


if __name__ == "__main__":
    process_all()
