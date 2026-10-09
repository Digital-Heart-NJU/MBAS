#!/usr/bin/env python3
"""Reconstruct a smooth, closed outer surface for one CARELA left atrium.

The wall generated in step 1 is a voxel shell. Extracting that shell directly
would produce both its inner and outer faces. Here the cleaned lumen and wall
are first combined into a solid, a signed-distance field (SDF) is built around
the solid, and the zero level set is extracted and smoothed. The resulting
watertight triangular surface can be used as the epicardial boundary of a later
tetrahedral wall reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
except ModuleNotFoundError:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from step01_geometry_fron_nifti import (
    DEFAULT_PARAMS,
    build_closed_wall,
    clean_lumen,
    load_json,
    load_lumen_mask,
    parse_atrium_label,
    read_nifti,
    resolve_params_from_data,
    resample_to_isotropic,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def validate_surface_parameters(
    sdf_sigma_mm: float,
    smooth_iterations: int,
    pass_band: float,
    crop_padding_mm: float,
) -> None:
    if sdf_sigma_mm < 0:
        raise ValueError("sdf_sigma_mm must be >= 0")
    if smooth_iterations < 0:
        raise ValueError("smooth_iterations must be >= 0")
    if not 0 < pass_band <= 2:
        raise ValueError("pass_band must be in the interval (0, 2]")
    if crop_padding_mm < 0:
        raise ValueError("crop_padding_mm must be >= 0")


def crop_with_padding(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    pad_mm: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Crop a mask while retaining a physical background margin.

    At least one background voxel is added on every side. If the object touches
    the source-image boundary, explicit zero padding is added and the returned
    offset is allowed to be negative. Marching cubes then has a sign change on
    every side and can produce a closed mesh.
    """
 
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, got shape {mask.shape}")
    if any(spacing <= 0 for spacing in spacing_zyx):
        raise ValueError(f"Spacing values must be positive, got {spacing_zyx}")

    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot crop an empty solid mask")

    pad = np.array(
        [max(1, int(math.ceil(pad_mm / spacing))) for spacing in spacing_zyx],
        dtype=int,
    )
    requested_lo = coords.min(axis=0) - pad
    requested_hi = coords.max(axis=0) + pad + 1
    source_lo = np.maximum(requested_lo, 0)
    source_hi = np.minimum(requested_hi, np.asarray(mask.shape))

    cropped = mask[
        source_lo[0] : source_hi[0],
        source_lo[1] : source_hi[1],
        source_lo[2] : source_hi[2],
    ]
    pad_before = source_lo - requested_lo
    pad_after = requested_hi - source_hi
    if np.any(pad_before) or np.any(pad_after):
        cropped = np.pad(
            cropped,
            tuple((int(before), int(after)) for before, after in zip(pad_before, pad_after)),
            mode="constant",
            constant_values=False,
        )

    return cropped, tuple(int(value) for value in requested_lo) 
  
  def shifted_origin_xyz(
    origin_xyz: tuple[float, float, float],
    offset_zyx: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
) -> tuple[float, float, float]:
    oz, oy, ox = offset_zyx
    sz, sy, sx = spacing_zyx
    return (
        float(origin_xyz[0] + ox * sx),
        float(origin_xyz[1] + oy * sy),
        float(origin_xyz[2] + oz * sz),
    )


def numpy_to_float_image(
    arr_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> vtk.vtkImageData:
    arr = np.ascontiguousarray(arr_zyx.astype(np.float32, copy=False))
    nz, ny, nx = arr.shape

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    sz, sy, sx = spacing_zyx
    image.SetSpacing(sx, sy, sz)
    image.SetOrigin(origin_xyz)

    vtk_arr = numpy_to_vtk(
        arr.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    vtk_arr.SetName("sdf")
    image.GetPointData().SetScalars(vtk_arr)
    return image

def largest_surface(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    if poly.GetNumberOfCells() == 0:
        raise ValueError("Marching cubes produced an empty surface")

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    output = vtk.vtkPolyData()
    output.DeepCopy(connectivity.GetOutput())
    return output


def normal_outward_fraction(poly: vtk.vtkPolyData) -> float | None:
    points = poly.GetPoints()
    normals = poly.GetPointData().GetNormals()
    if points is None or normals is None or poly.GetNumberOfPoints() == 0:
        return None

    point_values = vtk_to_numpy(points.GetData())
    normal_values = vtk_to_numpy(normals)
    center = point_values.mean(axis=0)
    dots = ((point_values - center) * normal_values).sum(axis=1)
    return float(np.mean(dots > 0))

def strip_display_scalars(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Keep normals but remove SDF/region arrays that alter mesh display."""

    output = vtk.vtkPolyData()
    output.DeepCopy(poly)
    point_data = output.GetPointData()
    normals = point_data.GetNormals()
    normal_name = normals.GetName() if normals is not None else None
    for index in reversed(range(point_data.GetNumberOfArrays())):
        if point_data.GetArrayName(index) != normal_name:
            point_data.RemoveArray(index)
    point_data.SetScalars(None)
    output.GetCellData().Initialize()
    return output

def orient_normals_outward(
    poly: vtk.vtkPolyData,
) -> tuple[vtk.vtkPolyData, bool, float | None, float | None]:
    """Make triangle winding and point normals consistently face outwards."""

    before = normal_outward_fraction(poly)
    flipped = False
    oriented = poly

    if before is not None and before < 0.5:
        reverse = vtk.vtkReverseSense()
        reverse.SetInputData(oriented)
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOn()
        reverse.Update()
        oriented = reverse.GetOutput()
        flipped = True

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(oriented)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()
    oriented = normals.GetOutput()

    after = normal_outward_fraction(oriented)
    if after is not None and after < 0.5:
        reverse = vtk.vtkReverseSense()
        reverse.SetInputData(oriented)
        reverse.ReverseCellsOn()
        reverse.ReverseNormalsOn()
        reverse.Update()
        oriented = reverse.GetOutput()
        flipped = True

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(oriented)
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOff()
        normals.SplittingOff()
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.Update()
        oriented = normals.GetOutput()
        after = normal_outward_fraction(oriented)

    return strip_display_scalars(oriented), flipped, before, after


def extract_sdf_outer_surface(
    solid_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    sdf_sigma_mm: float,
    smooth_iterations: int,
    pass_band: float,
    crop_padding_mm: float,
) -> vtk.vtkPolyData:
    validate_surface_parameters(
        sdf_sigma_mm,
        smooth_iterations,
        pass_band,
        crop_padding_mm,
    )
    crop, offset = crop_with_padding(solid_mask, spacing_zyx, crop_padding_mm)
    crop_origin = shifted_origin_xyz(origin_xyz, offset, spacing_zyx)

    outside = ndi.distance_transform_edt(~crop, sampling=spacing_zyx)
    inside = ndi.distance_transform_edt(crop, sampling=spacing_zyx)
    outside -= inside
    sdf = outside
    del inside

    sdf = sdf.astype(np.float32, copy=False)
    if sdf_sigma_mm > 0:
        sigma_vox = [sdf_sigma_mm / spacing for spacing in spacing_zyx]
        sdf = ndi.gaussian_filter(sdf, sigma=sigma_vox)

    image = numpy_to_float_image(sdf, spacing_zyx, crop_origin)
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(image)
    contour.SetValue(0, 0.0)
    contour.ComputeNormalsOn()
    contour.ComputeGradientsOff()
    contour.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(contour.GetOutputPort())
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()
    poly = largest_surface(clean.GetOutput())

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

        clean_after_smoothing = vtk.vtkCleanPolyData()
        clean_after_smoothing.SetInputConnection(smooth.GetOutputPort())
        clean_after_smoothing.Update()
        poly = largest_surface(clean_after_smoothing.GetOutput())

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()

    oriented, _, _, _ = orient_normals_outward(normals.GetOutput())
    return oriented


def write_surface(path: Path, poly: vtk.vtkPolyData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        writer = vtk.vtkOBJWriter()
    elif suffix == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetDataModeToBinary()
    elif suffix == ".stl":
        writer = vtk.vtkSTLWriter()
        writer.SetFileTypeToBinary()
    else:
        raise ValueError(f"Unsupported output surface extension: {path.suffix}")

    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    if writer.Write() != 1 or not path.is_file():
        raise OSError(f"Failed to write surface: {path}")


def mesh_qc(poly: vtk.vtkPolyData) -> dict[str, Any]:
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputData(poly)
    triangle_filter.Update()

    mass = vtk.vtkMassProperties()
    mass.SetInputConnection(triangle_filter.GetOutputPort())
    mass.Update()

    bounds = poly.GetBounds()
    boundary_edges = count_feature_edges(poly, boundary=True, non_manifold=False)
    non_manifold_edges = count_feature_edges(poly, boundary=False, non_manifold=True)
    return {
        "points": int(poly.GetNumberOfPoints()),
        "cells": int(poly.GetNumberOfCells()),
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "surface_area_mm2": float(mass.GetSurfaceArea()),
        "enclosed_volume_ml": float(mass.GetVolume() / 1000.0),
        "bounds_xyz_mm": [float(value) for value in bounds],
        "watertight_manifold": bool(boundary_edges == 0 and non_manifold_edges == 0),
    }


def reconstruct_closed_solid(
    case: str,
    root: Path,
    params: dict[str, Any],
) -> tuple[
    np.ndarray,
    tuple[float, float, float],
    tuple[float, float, float],
    dict[str, Any],
]:
    atrium_path = root / case / "atriumSegImgMO.nii.gz"
    if not atrium_path.exists():
        raise FileNotFoundError(atrium_path)

    atrium, meta = read_nifti(atrium_path)
    effective_params = resolve_params_from_data(params, atrium, meta)
    source_spacing = meta.spacing_zyx
    source_voxel_volume = meta.voxel_volume_mm3

    lumen, mask_source = load_lumen_mask(atrium, effective_params)
    lumen = clean_lumen(lumen, source_spacing, source_voxel_volume, effective_params)

    target_spacing_mm = float(effective_params["geometry_workspace_spacing_mm"])
    lumen_iso, iso_spacing, zoom_factors = resample_to_isotropic(
        lumen,
        source_spacing,
        target_spacing_mm,
    )
    iso_voxel_volume = float(np.prod(iso_spacing))
    lumen_iso = clean_lumen(lumen_iso, iso_spacing, iso_voxel_volume, effective_params)

    closed_wall = build_closed_wall(
        lumen_iso,
        iso_spacing,
        float(effective_params["wall_out_mm"]),
    )
    closed_solid = lumen_iso | closed_wall

    info = {
        "effective_params": effective_params,
        "mask_source": mask_source,
        "source_shape_zyx": [int(value) for value in atrium.shape],
        "solid_shape_zyx": [int(value) for value in closed_solid.shape],
        "source_spacing_zyx_mm": [float(value) for value in source_spacing],
        "solid_spacing_zyx_mm": [float(value) for value in iso_spacing],
        "origin_xyz_mm": [float(value) for value in meta.origin_xyz],
        "zoom_factors_zyx": [float(value) for value in zoom_factors],
        "lumen_voxels": int(lumen_iso.sum()),
        "closed_wall_voxels": int(closed_wall.sum()),
        "closed_solid_voxels": int(closed_solid.sum()),
        "lumen_volume_ml": float(lumen_iso.sum() * iso_voxel_volume / 1000.0),
        "closed_wall_volume_ml": float(closed_wall.sum() * iso_voxel_volume / 1000.0),
        "closed_solid_volume_ml": float(closed_solid.sum() * iso_voxel_volume / 1000.0),
    }
    return closed_solid, iso_spacing, meta.origin_xyz, info


def process_case(
    case: str,
    root: Path,
    params: dict[str, Any],
    mesh_root: Path,
    sdf_sigma_mm: float,
    smooth_iterations: int,
    pass_band: float,
    crop_padding_mm: float,
    output_format: str,
) -> dict[str, Any]:
    validate_surface_parameters(
        sdf_sigma_mm,
        smooth_iterations,
        pass_band,
        crop_padding_mm,
    )
    solid, spacing_zyx, origin_xyz, solid_info = reconstruct_closed_solid(
        case,
        root,
        params,
    )

    case_mesh_dir = mesh_root / case
    surface_file = case_mesh_dir / f"{case}_closed_wall_surface.{output_format}"
    params_file = case_mesh_dir / f"{case}_effective_params.json"
    summary_file = case_mesh_dir / f"{case}_step02_surface_summary.json"

    poly = extract_sdf_outer_surface(
        solid,
        spacing_zyx=spacing_zyx,
        origin_xyz=origin_xyz,
        sdf_sigma_mm=sdf_sigma_mm,
        smooth_iterations=smooth_iterations,
        pass_band=pass_band,
        crop_padding_mm=crop_padding_mm,
    )
    poly, normals_flipped, outward_before, outward_after = orient_normals_outward(poly)
    quality = mesh_qc(poly)
    if not quality["watertight_manifold"]:
        raise RuntimeError(
            "The reconstructed surface is not a watertight manifold: "
            f"boundary_edges={quality['boundary_edges']}, "
            f"non_manifold_edges={quality['non_manifold_edges']}"
        )

    write_surface(surface_file, poly)
    summary: dict[str, Any] = {
        "case": case,
        "surface_source": "zero_level_of_smoothed_sdf_from_closed_solid",
        "parameters": {
            "sdf_sigma_mm": float(sdf_sigma_mm),
            "smooth_iterations": int(smooth_iterations),
            "pass_band": float(pass_band),
            "crop_padding_mm": float(crop_padding_mm),
            "output_format": output_format,
        },
        "output_files": {
            "closed_wall_surface": str(surface_file),
            "effective_params": str(params_file),
            "summary": str(summary_file),
        },
        "solid_info": solid_info,
        "mesh": {
            "normals_flipped": normals_flipped,
            "outward_fraction_before_flip": outward_before,
            "outward_fraction_after_flip": outward_after,
            **quality,
        },
        "ready_as_epicardial_tetra_boundary": bool(quality["watertight_manifold"]),
    }
    write_json(params_file, solid_info["effective_params"])
    write_json(summary_file, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="train_1")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=Path("step02_outputs/closed_wall_surfaces"),
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON parameter file")

    parser.add_argument(
        "--wall-out-mm",
        type=float,
        default=None,
        help="Fixed outward wall thickness in mm (may also be supplied by --config)",
    )
    parser.add_argument("--close-radius-mm", type=float, default=None)
    parser.add_argument("--target-spacing-mm", type=float, default=None)
    parser.add_argument(
        "--atrium-label",
        default=None,
        help="Use auto, nonzero, or a numeric segmentation label",
    )
    parser.add_argument("--fill-holes", dest="fill_holes", action="store_true", default=None)
    parser.add_argument("--no-fill-holes", dest="fill_holes", action="store_false")

    parser.add_argument("--output-format", choices=("obj", "vtp", "stl"), default="obj")
    parser.add_argument("--sdf-sigma-mm", type=float, default=0.9)
    parser.add_argument("--smooth-iterations", type=int, default=80)
    parser.add_argument("--pass-band", type=float, default=0.025)
    parser.add_argument("--crop-padding-mm", type=float, default=8.0)
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    if args.config is not None:
        params.update(load_json(args.config.resolve()))

    if args.wall_out_mm is not None:
        params["wall_out_mm"] = args.wall_out_mm
    if args.close_radius_mm is not None:
        params["close_radius_mm"] = args.close_radius_mm
    if args.target_spacing_mm is not None:
        params["geometry_workspace_spacing_mm"] = args.target_spacing_mm
    if args.atrium_label is not None:
        params["atrium_label"] = parse_atrium_label(args.atrium_label)
    if args.fill_holes is not None:
        params["fill_holes"] = args.fill_holes

    summary = process_case(
        case=args.case,
        root=args.root.resolve(),
        params=params,
        mesh_root=args.mesh_root.resolve(),
        sdf_sigma_mm=args.sdf_sigma_mm,
        smooth_iterations=args.smooth_iterations,
        pass_band=args.pass_band,
        crop_padding_mm=args.crop_padding_mm,
        output_format=args.output_format,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()







