"""
Download real Micro-CT trabecular bone data.

PRIMARY:   Zenodo — 100x100x100 voxel trabecular bone samples (already 3D volumes)
SECONDARY: Figshare — larger MicroCT scans (152x152x432), need patch extraction

Run: python data/download_data.py
"""

import os
import zipfile
import tarfile
import requests
from tqdm import tqdm

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def download_file(url, dest_path, desc="Downloading"):
    """Stream-download a file with a progress bar."""
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_zenodo():
    """
    Zenodo record 2598306 — MicroCT Trabecular Bone Samples.
    100x100x100 voxel volumes at ~10 micron resolution.
    These are already segmented (bone=1, pore=0) — perfect for direct use.
    """
    print("\n=== Downloading Zenodo trabecular bone dataset ===")
    # Zenodo API to get file list
    record_id = "2598306"
    api_url = f"https://zenodo.org/api/records/{record_id}"

    try:
        r = requests.get(api_url, timeout=30)
        r.raise_for_status()
        files = r.json().get("files", [])

        if not files:
            print("No files found via API — using direct URL fallback.")
            _download_zenodo_direct()
            return

        for f in files[:5]:  # download first 5 samples
            fname = f["key"]
            furl = f["links"]["self"]
            dest = os.path.join(RAW_DIR, fname)
            if os.path.exists(dest):
                print(f"  Already exists: {fname}")
                continue
            download_file(furl, dest, desc=f"  {fname}")

        print("Zenodo download complete.")

    except Exception as e:
        print(f"Zenodo API failed ({e}) — trying direct download...")
        _download_zenodo_direct()


def _download_zenodo_direct():
    """
    Direct fallback URLs for Zenodo bone dataset files.
    These are the actual micro-CT NIfTI/numpy binary volumes.
    """
    files = [
        ("https://zenodo.org/record/2598306/files/sample_01.npy", "sample_01.npy"),
        ("https://zenodo.org/record/2598306/files/sample_02.npy", "sample_02.npy"),
        ("https://zenodo.org/record/2598306/files/sample_03.npy", "sample_03.npy"),
    ]
    for url, name in files:
        dest = os.path.join(RAW_DIR, name)
        if not os.path.exists(dest):
            try:
                download_file(url, dest, desc=f"  {name}")
            except Exception as e:
                print(f"  Failed: {name} — {e}")


def download_figshare():
    """
    Figshare — MicroCT scans of bone and cement-bone microstructures.
    Larger volumes (152x152x432 voxels) — we extract 64³ patches from these.
    """
    print("\n=== Downloading Figshare MicroCT dataset ===")

    # Figshare article API
    article_id = "5466397"  # bone microstructure dataset
    api_url = f"https://api.figshare.com/v2/articles/{article_id}/files"

    try:
        r = requests.get(api_url, timeout=30)
        r.raise_for_status()
        files = r.json()

        for f in files[:3]:
            fname = f["name"]
            furl = f["download_url"]
            dest = os.path.join(RAW_DIR, fname)
            if os.path.exists(dest):
                print(f"  Already exists: {fname}")
                continue
            download_file(furl, dest, desc=f"  {fname}")

        print("Figshare download complete.")

    except Exception as e:
        print(f"Figshare download failed: {e}")
        print("  Manual download: https://figshare.com/articles/dataset/5466397")
        print("  Place files in data/raw/")


if __name__ == "__main__":
    print("Synapse-Vis — Real Micro-CT Data Downloader")
    print("=" * 50)
    download_zenodo()
    download_figshare()
    print("\nAll downloads complete. Run: python data/preprocess.py")
