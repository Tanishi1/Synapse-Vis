"""
Geometry: soft-continuous voxel field → smoothed STL mesh.

KEY FIX: apply_cylinder_mask no longer adds cortical rings.
The scaffold is uniformly porous top-to-bottom, like a real clinical implant.
"""

import os
import numpy as np
from skimage.measure import marching_cubes
import trimesh


def apply_cylinder_mask(voxel_grid: np.ndarray) -> np.ndarray:
    """
    Carve the voxel grid into a clean cylinder — nothing else.

    REMOVED: cortical ring caps at top/bottom.
    Those lines were causing the solid ring artifacts in the viewer.
    A real bone scaffold implant is uniformly porous throughout.
    """
    masked = voxel_grid.copy()
    D, H, W = masked.shape

    z_idx, y_idx, x_idx = np.indices(masked.shape)
    cx, cz = W / 2.0, D / 2.0
    R_outer = min(W, D) / 2.0 - 1.5
    dist_xz = np.sqrt((x_idx - cx) ** 2 + (z_idx - cz) ** 2)

    is_soft = not _is_binary(voxel_grid)
    if is_soft:
        # Smooth sigmoid transition at the outer cylinder wall.
        # sharpness controls how many voxels the transition spans (~3–4 voxels).
        sharpness = 3.0
        # outside_weight: 0 inside cylinder, 1 outside
        outside_weight = (np.tanh((dist_xz - R_outer) * sharpness) + 1) / 2
        # Blend field toward 0.0 (pore) outside the cylinder
        masked = voxel_grid * (1 - outside_weight)
    else:
        # Binary field: hard cut
        masked[dist_xz > R_outer] = 0

    return masked


def voxel_to_stl(voxel_grid: np.ndarray, filepath: str, smooth: bool = True) -> tuple:
    """
    Convert voxel field → smoothed STL file.

    Soft continuous fields (trained VAE output): direct marching cubes at 0.5.
    Binary fields (legacy): Gaussian pre-blur + marching cubes.
    Taubin smoothing in both cases for organic bone-like surface curves.
    """
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if _is_binary(voxel_grid):
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(voxel_grid.astype(np.float32), sigma=1.5)
    else:
        field = voxel_grid.astype(np.float32)

    level = 0.5
    padded = np.pad(field, pad_width=2, mode="constant", constant_values=0.0)
    verts, faces, normals, _ = marching_cubes(padded, level=level)

    if len(faces) == 0:
        raise ValueError("No surface found — check field values around 0.5")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

    if smooth:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=30)

    mesh.apply_translation(-mesh.centroid)
    current_size = max(mesh.extents)
    if current_size > 0:
        mesh.apply_scale(15.0 / current_size)

    mesh.export(filepath, file_type="stl")
    return filepath, len(mesh.faces)


def _is_binary(grid: np.ndarray) -> bool:
    flat = grid.ravel()
    sample = flat[::max(1, len(flat) // 2000)]
    unique = np.unique(np.round(sample, 1))
    return len(unique) <= 3 and set(unique).issubset({0.0, 0.1, 0.9, 1.0})
