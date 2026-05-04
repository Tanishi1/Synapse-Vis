"""
Geometry conversion: binary voxel grid → smoothed 3D mesh → STL file.

Pipeline:
  1. Marching Cubes — extracts a triangle mesh from the isosurface
     of the binary voxel grid at level=0.5
  2. Laplacian smoothing — removes the "staircase" artifact caused
     by the cubic voxel grid, rounds edges without changing topology
  3. STL export — universal 3D printing format accepted by all printers
"""

import os
import numpy as np
from skimage.measure import marching_cubes
import trimesh


def voxel_to_stl(voxel_grid: np.ndarray, filepath: str, smooth: bool = True) -> tuple:
    """
    Convert a binary voxel grid to a smoothed STL mesh file.

    Args:
        voxel_grid: binary numpy array (64, 64, 64), 1=solid, 0=pore
        filepath: output path for the STL file
        smooth: whether to apply Laplacian smoothing (recommended)

    Returns:
        (filepath, face_count) — confirms file was written
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Pad the volume by 1 voxel on each side with solid material.
    # This ensures Marching Cubes produces a closed mesh (no open edges)
    # which is required for valid STL files.
    padded = np.pad(voxel_grid, pad_width=1, mode="constant", constant_values=0)

    # Marching Cubes: find the isosurface at the solid/void boundary.
    # level=0.5 is the midpoint between 0 (void) and 1 (solid).
    # Returns: vertices (Nx3), faces (Mx3 indices), normals, values
    verts, faces, normals, _ = marching_cubes(padded, level=0.5)

    if len(faces) == 0:
        raise ValueError(
            "Marching Cubes returned no faces. The voxel grid may be "
            "entirely solid or entirely empty. Check generation threshold."
        )

    # Build mesh
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=True,  # remove duplicate vertices, fix winding
    )

    # Laplacian smoothing: moves each vertex toward the average position
    # of its neighbours. 2-3 iterations is enough to remove staircase
    # artifacts without losing structural detail.
    if smooth:
        trimesh.smoothing.filter_laplacian(mesh, iterations=3)

    # Center the mesh at the origin so the Three.js viewer can frame it
    mesh.apply_translation(-mesh.centroid)

    # Scale to a realistic implant size (~15mm across the longest axis).
    # This is just for visualization — the real printer would use the
    # actual patient-specific dimensions.
    target_size_mm = 15.0
    current_size = max(mesh.extents)
    if current_size > 0:
        scale = target_size_mm / current_size
        mesh.apply_scale(scale)

    # Export as binary STL (smaller file than ASCII STL)
    mesh.export(filepath, file_type="stl")

    return filepath, len(mesh.faces)
