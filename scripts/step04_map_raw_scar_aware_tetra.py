#!/usr/bin/env python3
"""Map raw expert scar labels onto an existing tetrahedral LA wall mesh.

This is intentionally a strict mapping step. The expert scar mask is resampled
to the same isotropic geometry space with nearest-neighbour interpolation, then
tetrahedra are labelled using direct point samples in the raw scar volume.
There is no dilation, wall projection, nearest-wall snapping, or morphology
expansion. The report explicitly records any mismatch between the raw scar and
the generated myocardium wall.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
except ModuleNotFoundError:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

try:
    from step01_geometry_from_nifti import (
        build_closed_wall,
        build_open_wall_candidate,
        clean_lumen,
        infer_distal_exclusion,
        load_json,
        load_lumen_mask,
        make_isotropic_meta,
        read_nifti,
        resolve_params_from_data,
        resample_to_isotropic,
        write_nifti,
    )
except ModuleNotFoundError:
    from step01_geometry_fron_nifti import (
        build_closed_wall,
        build_open_wall_candidate,
        clean_lumen,
        infer_distal_exclusion,
        load_json,
        load_lumen_mask,
        make_isotropic_meta,
        read_nifti,
        resolve_params_from_data,
        resample_to_isotropic,
        write_nifti,
    )

from step02_surface_reconstruction import (
    count_feature_edges,
    orient_normals_outward,
    write_surface,
)
from step03_build_tetra_mesh import read_elem, read_pts, resolve_config_path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

def percentile_stats(values:np.ndarray,unit:str | None = None) -> dict[str,Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        result: dict[str, Any] = {"count": 0}
    else:
        result = {
            "count": int(values.size),
            "min": float(np.min(values)),
            "p01": float(np.percentile(values, 1)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.percentile(values, 50)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }
    if unit is not None:
        result["unit"] = unit
    return result


def resolve_geometry_params(config: dict[str, Any], atrium: np.ndarray, meta: Any) -> dict[str, Any]:
    """Resolve auto parameters while preserving the opening section."""

    params = resolve_params_from_data(config, atrium, meta)
    if "auto_opening" in config:
        params["auto_opening"] = config["auto_opening"]
    return params

def reconstruct_wall_masks(
    case: str,
    root: Path,
    config: dict[str, Any],
    version: str,
) -> tuple[
    dict[str, np.ndarray],
    tuple[float, float, float],
    tuple[float, float, float],
    Any,
    dict[str, Any],
]:
    atrium_path = root / case / "atriumSegImgMO.nii.gz"
    if not atrium_path.is_file():
        raise FileNotFoundError(atrium_path)

    atrium, meta = read_nifti(atrium_path)
    params = resolve_geometry_params(config, atrium, meta)
    lumen, mask_source = load_lumen_mask(atrium, params)
    lumen = clean_lumen(lumen, meta.spacing_zyx, meta.voxel_volume_mm3, params)

    target_spacing_mm = float(params["geometry_workspace_spacing_mm"])
    lumen_iso, iso_spacing, zoom_factors = resample_to_isotropic(
        lumen, meta.spacing_zyx, target_spacing_mm
    )
    iso_voxel_volume = float(np.prod(iso_spacing))
    lumen_iso = clean_lumen(lumen_iso, iso_spacing, iso_voxel_volume, params)
    closed_wall = build_closed_wall(lumen_iso, iso_spacing, float(params["wall_out_mm"]))

    opening_info: dict[str, Any] | None = None
    if version == "closed":
        retained_lumen = lumen_iso
        wall = closed_wall
    elif version == "open_candidate":
        exclusion, opening_info = infer_distal_exclusion(
            lumen_iso, iso_spacing, iso_voxel_volume, params
        )
        retained_lumen = lumen_iso & ~exclusion
        wall = build_open_wall_candidate(closed_wall, exclusion, iso_spacing, params)
    else:
        raise ValueError(f"Unsupported version: {version!r}")

    iso_meta = make_isotropic_meta(meta, wall, target_spacing_mm)
    info = {
        "case": case,
        "version": version,
        "effective_params": params,
        "mask_source": mask_source,
        "source_shape_zyx": [int(value) for value in atrium.shape],
        "source_spacing_zyx_mm": [float(value) for value in meta.spacing_zyx],
        "iso_shape_zyx": [int(value) for value in lumen_iso.shape],
        "iso_spacing_zyx_mm": [float(value) for value in iso_spacing],
        "origin_xyz_mm": [float(value) for value in meta.origin_xyz],
        "zoom_factors_zyx": [float(value) for value in zoom_factors],
        "retained_lumen_voxels": int(retained_lumen.sum()),
        "wall_voxels": int(wall.sum()),
        "wall_ml": float(wall.sum() * iso_voxel_volume / 1000.0),
        "opening_inference": opening_info,
    }
    return {"lumen": retained_lumen, "wall": wall}, iso_spacing, meta.origin_xyz, iso_meta, info


def load_raw_scar_iso(
    case: str,
    root: Path,
    target_spacing_mm: float,
    scar_label_min: float,
) -> tuple[np.ndarray, tuple[float, float, float], dict[str, Any]]:
    scar_path = root / case / "scarSegImgM.nii.gz"
    if not scar_path.is_file():
        raise FileNotFoundError(scar_path)

    scar, meta = read_nifti(scar_path)
    scar_mask = scar >= scar_label_min
    scar_iso, iso_spacing, zoom_factors = resample_to_isotropic(
        scar_mask, meta.spacing_zyx, target_spacing_mm
    )
    info = {
        "source_file": str(scar_path),
        "source_voxels": int(scar_mask.sum()),
        "source_ml": float(scar_mask.sum() * meta.voxel_volume_mm3 / 1000.0),
        "source_shape_zyx": [int(value) for value in scar.shape],
        "source_spacing_zyx_mm": [float(value) for value in meta.spacing_zyx],
        "source_origin_xyz_mm": [float(value) for value in meta.origin_xyz],
        "isotropic_voxels": int(scar_iso.sum()),
        "isotropic_spacing_zyx_mm": [float(value) for value in iso_spacing],
        "isotropic_shape_zyx": [int(value) for value in scar_iso.shape],
        "isotropic_origin_xyz_mm": [float(value) for value in meta.origin_xyz],
        "isotropic_ml": float(scar_iso.sum() * np.prod(iso_spacing) / 1000.0),
        "zoom_factors_zyx": [float(value) for value in zoom_factors],
        "scar_label_min": float(scar_label_min),
        "mapping_principle": "raw expert scar; nearest-neighbor resampling only; no projection; no dilation",
    }
    return scar_iso, meta.origin_xyz, info


def mask_to_image(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> vtk.vtkImageData:
    arr = np.ascontiguousarray(mask.astype(np.uint8, copy=False))
    nz, ny, nx = arr.shape
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    sz, sy, sx = spacing_zyx
    image.SetSpacing(sx, sy, sz)
    image.SetOrigin(origin_xyz)
    vtk_arr = numpy_to_vtk(
        arr.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_arr.SetName("label")
    image.GetPointData().SetScalars(vtk_arr)
    return image


def mask_to_surface(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    smooth_iterations: int,
    pass_band: float,
) -> vtk.vtkPolyData:
    if not mask.any():
        raise ValueError("Cannot create a scar surface from an empty mask")
    image = mask_to_image(mask, spacing_zyx, origin_xyz)
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(image)
    contour.SetValue(0, 1)
    contour.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(contour.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    poly = clean.GetOutput()
    if poly.GetNumberOfCells() == 0:
        raise ValueError("Discrete marching cubes produced an empty scar surface")

    if smooth_iterations > 0:
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputData(poly)
        smooth.SetNumberOfIterations(smooth_iterations)
        smooth.SetPassBand(pass_band)
        smooth.BoundarySmoothingOff()
        smooth.FeatureEdgeSmoothingOff()
        smooth.NonManifoldSmoothingOff()
        smooth.NormalizeCoordinatesOn()
        smooth.Update()
        poly = smooth.GetOutput()

    oriented, _, _, _ = orient_normals_outward(poly)
    return oriented






