"""
Simplified bone ingrowth diffusion simulation.

Models bone tissue growth through the scaffold's pore network
using iterative morphological dilation + connectivity constraints.

This is NOT a real biological simulation — it's a diffusion model
that demonstrates the principle: connected pores = bone grows through,
disconnected pores = bone growth stops at dead ends.

Output: a sequence of voxel grids showing bone filling the scaffold
over N time steps (each step = ~1 month of simulated healing).
"""

import numpy as np
from scipy import ndimage


def simulate_ingrowth(scaffold_grid: np.ndarray, n_steps: int = 20) -> list:
    """
    Simulate bone ingrowth through the scaffold pore network.

    Args:
        scaffold_grid: binary (64,64,64), 1=solid scaffold, 0=pore
        n_steps: number of simulation steps (each ≈ 1 month)

    Returns:
        list of dicts, one per step:
          {
            "step": int,
            "bone_fill_pct": float,    # % of pore space filled with new bone
            "new_bone_grid": np.array,  # binary grid of new bone tissue
          }
    """
    scaffold = scaffold_grid.astype(np.float32)
    # Only voxels that are exactly 0 are valid pores. -1 is empty air outside the implant.
    pore_mask = scaffold == 0

    total_pore_voxels = np.sum(pore_mask)
    if total_pore_voxels == 0:
        return [{"step": 0, "bone_fill_pct": 0.0}]

    # Find the connected pore network (only connected pores get bone)
    labeled_pores, n_components = ndimage.label(pore_mask)
    component_sizes = np.array(
        ndimage.sum(pore_mask, labeled_pores, range(1, n_components + 1))
    )
    # Only grow bone in the largest connected component
    if len(component_sizes) > 0:
        largest_id = np.argmax(component_sizes) + 1
        connected_pores = labeled_pores == largest_id
    else:
        connected_pores = pore_mask

    # Seed points: pore voxels on the edges of the volume
    # (simulates bone growing in from the surrounding natural bone)
    seeds = np.zeros_like(scaffold, dtype=bool)
    seeds[0, :, :] = True
    seeds[-1, :, :] = True
    seeds[:, 0, :] = True
    seeds[:, -1, :] = True
    seeds[:, :, 0] = True
    seeds[:, :, -1] = True
    # Seeds must be in connected pore space
    seeds = seeds & connected_pores

    # Initialize bone growth front
    new_bone = seeds.copy()

    # Structuring element for 3D dilation (6-connected neighborhood)
    struct = ndimage.generate_binary_structure(3, 1)

    results = []

    for step in range(n_steps):
        # Dilate the bone front by 1 voxel
        expanded = ndimage.binary_dilation(new_bone, structure=struct, iterations=1)

        # Bone can only grow into connected pore space
        expanded = expanded & connected_pores

        # Add some stochasticity — not all adjacent voxels fill simultaneously
        # Random mask: 60-80% of eligible voxels actually fill each step
        if step > 0:
            random_mask = np.random.random(expanded.shape) < (0.6 + step * 0.01)
            expanded = expanded & (new_bone | (expanded & random_mask))

        new_bone = expanded

        bone_voxels = np.sum(new_bone)
        fill_pct = float(bone_voxels / total_pore_voxels * 100)

        results.append({
            "step": step + 1,
            "month": step + 1,
            "bone_fill_pct": round(fill_pct, 1),
            "new_bone_voxels": int(bone_voxels),
            "total_pore_voxels": int(total_pore_voxels),
        })

        # If we've filled 95%+ of connected pores, stop early
        connected_pore_count = np.sum(connected_pores)
        if bone_voxels >= connected_pore_count * 0.95:
            # Fill remaining steps with final state
            for s in range(step + 1, n_steps):
                results.append({
                    "step": s + 1,
                    "month": s + 1,
                    "bone_fill_pct": round(fill_pct, 1),
                    "new_bone_voxels": int(bone_voxels),
                    "total_pore_voxels": int(total_pore_voxels),
                })
            break

    return results


def get_ingrowth_at_step(scaffold_grid: np.ndarray, target_step: int) -> np.ndarray:
    """
    Get the bone growth state at a specific time step.
    Returns a grid where: 0=pore, 1=scaffold, 2=new bone.

    This is used to generate colored STL meshes showing bone fill.
    """
    scaffold = scaffold_grid.astype(np.float32)
    pore_mask = scaffold < 0.5

    labeled_pores, n_components = ndimage.label(pore_mask)
    component_sizes = np.array(
        ndimage.sum(pore_mask, labeled_pores, range(1, n_components + 1))
    )
    if len(component_sizes) > 0:
        largest_id = np.argmax(component_sizes) + 1
        connected_pores = labeled_pores == largest_id
    else:
        connected_pores = pore_mask

    # Seeds at volume boundary
    seeds = np.zeros_like(scaffold, dtype=bool)
    seeds[0, :, :] = True; seeds[-1, :, :] = True
    seeds[:, 0, :] = True; seeds[:, -1, :] = True
    seeds[:, :, 0] = True; seeds[:, :, -1] = True
    seeds = seeds & connected_pores

    new_bone = seeds.copy()
    struct = ndimage.generate_binary_structure(3, 1)

    np.random.seed(42)  # deterministic for consistent visualization

    for step in range(target_step):
        expanded = ndimage.binary_dilation(new_bone, structure=struct, iterations=1)
        expanded = expanded & connected_pores
        if step > 0:
            random_mask = np.random.random(expanded.shape) < (0.6 + step * 0.01)
            expanded = expanded & (new_bone | (expanded & random_mask))
        new_bone = expanded

    # Composite grid: 0=empty pore, 1=scaffold, 2=new bone
    result = scaffold.copy()
    result[new_bone] = 2.0
    return result
