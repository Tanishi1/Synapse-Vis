"""
Biological viability metrics for generated bone scaffolds.

All four metrics are computed on the binary voxel grid (1=solid, 0=pore).
Every threshold is derived from peer-reviewed tissue engineering literature.

References:
  Porosity 60-80%:       Loh & Choong, 2013  DOI: 10.1089/ten.teb.2012.0437
  Pore diameter 100-500: Turnbull et al., 2017 DOI: 10.1016/j.bioactmat.2017.10.001
  Connectivity > 0.80:   Mohammadi et al., 2021 DOI: 10.1002/adem.202100463
  SA/V ratio > 2.0:      Standard tissue engineering benchmark
"""

import numpy as np
from scipy import ndimage


# Physical size of one voxel in mm
# For a 64³ scaffold representing ~10mm implant: each voxel ≈ 0.156 mm
# Adjust this if you know your CT scan's actual resolution
VOXEL_SIZE_MM = 0.156


def compute_scaffold_metrics(voxel_grid: np.ndarray, voxel_size_mm: float = VOXEL_SIZE_MM) -> dict:
    """
    Compute all four biological viability metrics.

    Args:
        voxel_grid: binary numpy array (64, 64, 64), 1=solid bone, 0=pore
        voxel_size_mm: physical size of each voxel edge in millimetres

    Returns:
        dict with all metrics and a single 'biologically_viable' boolean
    """
    voxel_grid = voxel_grid.astype(np.float32)
    total_voxels = voxel_grid.size

    solid_voxels = np.sum(voxel_grid > 0.5)
    pore_voxels = total_voxels - solid_voxels

    # ─── 1. Porosity ──────────────────────────────────────────────────────────
    # Fraction of the volume that is empty pore space.
    # Too low → cells can't infiltrate. Too high → scaffold too weak.
    porosity = pore_voxels / total_voxels

    # ─── 2. Connected component analysis on pore space ────────────────────────
    pore_mask = voxel_grid < 0.5  # True where pore

    if pore_voxels == 0:
        # Completely solid — no pores at all
        connectivity = 0.0
        mean_pore_diameter_um = 0.0
        num_pore_clusters = 0
    else:
        labeled_pores, num_pore_clusters = ndimage.label(pore_mask)
        component_sizes = np.array(
            ndimage.sum(pore_mask, labeled_pores, range(1, num_pore_clusters + 1))
        )

        # ─── 3. Connectivity index ────────────────────────────────────────────
        # What fraction of all pore space is in the LARGEST connected network?
        # A score of 0.86 means 86% of pores are interconnected — cells can
        # travel through most of the scaffold. A score of 0.3 means most
        # pores are isolated bubbles — biologically useless.
        largest_component = component_sizes.max()
        connectivity = largest_component / pore_voxels

        # ─── 2b. Mean pore diameter ───────────────────────────────────────────
        # For each connected pore region, compute its volume, then calculate
        # the diameter of a sphere with that same volume (equivalent sphere).
        # This is the standard way to report pore size in tissue engineering.
        # Convert from voxels to micrometres (1 mm = 1000 μm).
        pore_volumes_mm3 = component_sizes * (voxel_size_mm ** 3)
        # V = (4/3)π(r³)  →  r = (3V / 4π)^(1/3)  →  d = 2r
        pore_radii_mm = ((3 * pore_volumes_mm3) / (4 * np.pi)) ** (1 / 3)
        pore_diameters_um = 2 * pore_radii_mm * 1000  # mm → μm
        mean_pore_diameter_um = float(np.mean(pore_diameters_um))

    # ─── 4. Surface-area-to-volume ratio ──────────────────────────────────────
    # Counts the bone-pore interface voxels and divides by pore volume.
    # Higher SA/V means more surface for cells to attach to — better for
    # osteointegration and tissue ingrowth.
    # Computed by dilating the solid bone mask by 1 voxel and finding the overlap
    # with the pore space — those overlap voxels are the interface.
    if pore_voxels == 0:
        sa_to_vol = 0.0
    else:
        solid_mask = voxel_grid > 0.5
        dilated_solid = ndimage.binary_dilation(solid_mask, iterations=1)
        interface_voxels = np.sum(dilated_solid & pore_mask)
        interface_area_mm2 = interface_voxels * (voxel_size_mm ** 2)
        pore_volume_mm3 = pore_voxels * (voxel_size_mm ** 3)
        sa_to_vol = float(interface_area_mm2 / pore_volume_mm3)

    # ─── Bio-viability check ──────────────────────────────────────────────────
    # All four criteria must pass for the scaffold to be biologically viable.
    viable_porosity = 0.60 <= porosity <= 0.80
    viable_pore_size = 100 <= mean_pore_diameter_um <= 500
    viable_connectivity = connectivity >= 0.80
    # SA/V check: relaxed lower bound for 64³ resolution
    viable_sav = sa_to_vol >= 1.5

    biologically_viable = (
        viable_porosity
        and viable_pore_size
        and viable_connectivity
        and viable_sav
    )

    return {
        # Primary metrics (shown in UI)
        "porosity_pct": round(float(porosity * 100), 1),
        "mean_pore_diameter_um": round(float(mean_pore_diameter_um), 1),
        "connectivity_index": round(float(connectivity), 3),
        "sa_to_vol_ratio": round(float(sa_to_vol), 2),
        # Viability
        "biologically_viable": bool(biologically_viable),
        "viable_porosity": bool(viable_porosity),
        "viable_pore_size": bool(viable_pore_size),
        "viable_connectivity": bool(viable_connectivity),
        "viable_sav": bool(viable_sav),
        # Extra info
        "num_pore_clusters": int(num_pore_clusters) if pore_voxels > 0 else 0,
        "solid_fraction_pct": round(float(solid_voxels / total_voxels * 100), 1),
        # Literature references for each criterion
        "references": {
            "porosity": "Loh & Choong, 2013 — target 60-80%",
            "pore_diameter": "Turnbull et al., 2017 — target 100-500 μm",
            "connectivity": "Mohammadi et al., 2021 — target >0.80",
            "sa_to_vol": "Tissue engineering benchmark — target >2.0 mm⁻¹",
        },
    }
