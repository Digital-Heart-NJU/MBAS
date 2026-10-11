#!/usr/bin/env python3
"""Build a scar-aware LA wall mesh and map raw scar fractions.

This experimental branch uses plausible raw scar voxels as constraints on the
generated wall/lumen geometry. The raw expert mask itself is never dilated,
projected to the wall, or snapped to a surface. After remeshing, the unchanged
raw scar volume is sampled at barycentric points of each tetrahedron to produce
continuous ``scar_fraction`` values and healthy/core/border tags.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
        clean_lumen,
        component_stats,
        ellipsoid_structure,
        keep_largest_component,
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
        clean_lumen,
        component_stats,
        ellipsoid_structure,
        keep_largest_component,
        load_json,
        load_lumen_mask,
        make_isotropic_meta,
        read_nifti,
        resolve_params_from_data,
        resample_to_isotropic,
        write_nifti,
    )

from step02_surface_reconstruction import count_feature_edges, write_surface
from step03_build_tetra_mesh import (
    build_boundary_surfaces,
    percentile_stats,
    read_elem,
    read_pts,
    resolve_config_path,
    run_meshtool_generate,
    run_meshtool_quality,
    tetra_quality,
)
from step04_map_raw_scar_to_tetra import (
    generate_default_fibers,
    load_raw_scar_iso,
    mask_to_surface,
    write_scalar_dat,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def resolve_geometry_params(config: dict[str, Any], atrium: np.ndarray, meta: Any) -> dict[str, Any]:
    """Resolve automatic geometry parameters while retaining auto-opening."""

    params = resolve_params_from_data(config, atrium, meta)
    if "auto_opening" in config:
        params["auto_opening"] = config["auto_opening"]
    return params


def signed_distance_to_lumen(
    lumen: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    outside = ndi.distance_transform_edt(~lumen, sampling=spacing_zyx)
    inside = ndi.distance_transform_edt(lumen, sampling=spacing_zyx)
    signed = outside.astype(np.float32, copy=False)
    signed[lumen] = -inside[lumen]
    return signed


def overlap_fraction(numer: np.ndarray, denom: np.ndarray) -> float | None:
    denominator_count = int(np.count_nonzero(denom))
    if denominator_count == 0:
        return None
    return float(np.count_nonzero(numer & denom) / denominator_count)


def mask_stats(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> dict[str, Any]:
    voxel_volume = float(np.prod(spacing_zyx))
    _, components = component_stats(mask, voxel_volume)
    return {
        "voxels": int(np.count_nonzero(mask)),
        "ml": float(np.count_nonzero(mask) * voxel_volume / 1000.0),
        "component_count": len(components),
        "largest_component_ml": float(components[0]["ml"]) if components else 0.0,
    }


def build_scar_aware_masks(
    lumen: np.ndarray,
    raw_scar: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    wall_out_mm: float,
    scar_inner_mm: float,
    scar_outer_mm: float,
    geometry_margin_mm: float,
    bridge_mm: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if lumen.shape != raw_scar.shape:
        raise ValueError(f"Lumen shape {lumen.shape} does not match scar shape {raw_scar.shape}")
    if any(value < 0 for value in (scar_inner_mm, scar_outer_mm, geometry_margin_mm, bridge_mm)):
        raise ValueError("Scar geometry distances must be non-negative")

    voxel_volume = float(np.prod(spacing_zyx))
    base_wall = build_closed_wall(lumen, spacing_zyx, wall_out_mm)
    base_outer = lumen | base_wall
    signed = signed_distance_to_lumen(lumen, spacing_zyx)

    plausible_scar = raw_scar & (signed >= -scar_inner_mm) & (signed <= scar_outer_mm)
    implausible_scar = raw_scar & ~plausible_scar
    if geometry_margin_mm > 0:
        support = ndi.binary_dilation(
            plausible_scar,
            structure=ellipsoid_structure(geometry_margin_mm, spacing_zyx),
        )
        support &= signed >= -(scar_inner_mm + geometry_margin_mm)
        support &= signed <= scar_outer_mm + geometry_margin_mm
    else:
        support = plausible_scar.copy()

    bridge = np.zeros_like(lumen, dtype=bool)
    if bridge_mm > 0 and np.any(support):
        dist_to_base_outer = ndi.distance_transform_edt(~base_outer, sampling=spacing_zyx)
        dist_to_support = ndi.distance_transform_edt(~support, sampling=spacing_zyx)
        bridge = (dist_to_base_outer <= bridge_mm) & (dist_to_support <= bridge_mm)
        bridge &= signed >= 0
        bridge &= signed <= scar_outer_mm + bridge_mm + geometry_margin_mm

    scar_aware_outer = keep_largest_component(base_outer | support | bridge, voxel_volume)
    included_support = support & scar_aware_outer
    # Geometry correction only: raw_scar remains unchanged for final sampling.
    scar_aware_lumen = keep_largest_component(lumen & ~included_support, voxel_volume)
    if not np.any(scar_aware_lumen):
        raise ValueError("Scar-aware lumen became empty; reduce geometry constraints")
    scar_aware_outer |= scar_aware_lumen
    scar_aware_wall = scar_aware_outer & ~scar_aware_lumen

    base_metrics = {
        "base_lumen": mask_stats(lumen, spacing_zyx),
        "base_wall": mask_stats(base_wall, spacing_zyx),
        "base_outer_solid": mask_stats(base_outer, spacing_zyx),
        "raw_scar_inside_base_wall_fraction": overlap_fraction(base_wall, raw_scar),
        "raw_scar_inside_base_lumen_fraction": overlap_fraction(lumen, raw_scar),
        "raw_scar_outside_base_lumen_and_wall_fraction": overlap_fraction(~base_outer, raw_scar),
    }
    aware_metrics = {
        "scar_aware_lumen": mask_stats(scar_aware_lumen, spacing_zyx),
        "scar_aware_wall": mask_stats(scar_aware_wall, spacing_zyx),
        "scar_aware_outer_solid": mask_stats(scar_aware_outer, spacing_zyx),
        "raw_scar_inside_scar_aware_wall_fraction": overlap_fraction(scar_aware_wall, raw_scar),
        "raw_scar_inside_scar_aware_lumen_fraction": overlap_fraction(scar_aware_lumen, raw_scar),
        "raw_scar_outside_scar_aware_lumen_and_wall_fraction": overlap_fraction(
            ~(scar_aware_lumen | scar_aware_wall), raw_scar
        ),
    }
    info = {
        "principle": "Raw scar is not projected or dilated for mapping; plausible locations only constrain geometry.",
        "parameters": {
            "wall_out_mm": float(wall_out_mm),
            "scar_inner_mm": float(scar_inner_mm),
            "scar_outer_mm": float(scar_outer_mm),
            "geometry_margin_mm": float(geometry_margin_mm),
            "bridge_mm": float(bridge_mm),
        },
        "raw_scar_filtering": {
            "raw_scar_voxels": int(raw_scar.sum()),
            "plausible_scar_voxels": int(plausible_scar.sum()),
            "implausible_scar_voxels": int(implausible_scar.sum()),
            "plausible_fraction": overlap_fraction(plausible_scar, raw_scar),
            "signed_distance_all_raw_scar_mm": percentile_stats(signed[raw_scar], "mm"),
            "signed_distance_plausible_raw_scar_mm": percentile_stats(signed[plausible_scar], "mm"),
        },
        "geometry_constraints": {
            "support": mask_stats(support, spacing_zyx),
            "included_support": mask_stats(included_support, spacing_zyx),
            "bridge": mask_stats(bridge, spacing_zyx),
            "lumen_removed_by_constraint_voxels": int(np.count_nonzero(lumen & ~scar_aware_lumen)),
            "lumen_removed_by_constraint_ml": float(np.count_nonzero(lumen & ~scar_aware_lumen) * voxel_volume / 1000.0),
            "outer_added_voxels": int(np.count_nonzero(scar_aware_outer & ~base_outer)),
            "outer_added_ml": float(np.count_nonzero(scar_aware_outer & ~base_outer) * voxel_volume / 1000.0),
        },
        "base_geometry": base_metrics,
        "scar_aware_geometry": aware_metrics,
    }
    return {
        "lumen": scar_aware_lumen,
        "wall": scar_aware_wall,
        "outer": scar_aware_outer,
        "base_lumen": lumen,
        "base_wall": base_wall,
        "base_outer": base_outer,
        "raw_scar": raw_scar,
        "plausible_scar": plausible_scar,
        "support": support,
        "bridge": bridge,
    }, info


def barycentric_weights(order: int) -> np.ndarray:
    if order < 1:
        raise ValueError("sample_order must be >= 1")
    weights = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            for k in range(order + 1 - i - j):
                l = order - i - j - k
                weights.append((i / order, j / order, k / order, l / order))
    result = np.asarray(weights, dtype=np.float32)
    centroid = np.asarray([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
    if not np.any(np.all(np.isclose(result, centroid), axis=1)):
        result = np.vstack([centroid, result])
    return result


def sample_binary_mask_at_points(
    mask: np.ndarray,
    points_xyz: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    ox, oy, oz = origin_xyz
    coordinates = np.vstack([
        (points_xyz[:, 2] - oz) / sz,
        (points_xyz[:, 1] - oy) / sy,
        (points_xyz[:, 0] - ox) / sx,
    ])
    values = ndi.map_coordinates(
        mask.astype(np.float32), coordinates, order=0, mode="constant", cval=0.0
    )
    return values >= 0.5


def tetra_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    tetra_points = points[tets]
    signed = np.einsum(
        "ij,ij->i",
        np.cross(tetra_points[:, 1] - tetra_points[:, 0], tetra_points[:, 2] - tetra_points[:, 0]),
        tetra_points[:, 3] - tetra_points[:, 0],
    ) / 6.0
    return np.abs(signed)


def sample_scar_fraction(
    points: np.ndarray,
    tets: np.ndarray,
    scar_mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    sample_order: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("sample_chunk_size must be positive")
    weights = barycentric_weights(sample_order)
    fractions = np.zeros(tets.shape[0], dtype=np.float32)
    centroid_inside = np.zeros(tets.shape[0], dtype=bool)
    for start in range(0, tets.shape[0], chunk_size):
        end = min(start + chunk_size, tets.shape[0])
        tetra_points = points[tets[start:end]]
        samples = np.einsum("sf,nfc->nsc", weights, tetra_points)
        inside = sample_binary_mask_at_points(
            scar_mask, samples.reshape((-1, 3)), spacing_zyx, origin_xyz
        ).reshape(samples.shape[:2])
        fractions[start:end] = inside.mean(axis=1)
        centroid_inside[start:end] = sample_binary_mask_at_points(
            scar_mask, tetra_points.mean(axis=1), spacing_zyx, origin_xyz
        )
    return fractions, centroid_inside, {
        "method": "direct raw scar volume sampling on tetrahedra",
        "sample_order": int(sample_order),
        "sample_points_per_tetra": int(weights.shape[0]),
        "raw_scar_fraction_stats": percentile_stats(fractions),
        "centroid_inside_count": int(centroid_inside.sum()),
    }


def classify_fraction_tags(
    fractions: np.ndarray,
    centroid_inside: np.ndarray,
    border_min_fraction: float,
    core_min_fraction: float,
    healthy_tag: int,
    scar_core_tag: int,
    border_zone_tag: int,
    promote_centroid_to_core: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0 <= border_min_fraction <= core_min_fraction <= 1:
        raise ValueError("Require 0 <= border_min_fraction <= core_min_fraction <= 1")
    if len({healthy_tag, scar_core_tag, border_zone_tag}) != 3:
        raise ValueError("healthy, core, and border tags must be distinct")
    core = fractions >= core_min_fraction
    if promote_centroid_to_core:
        core |= centroid_inside
    border = (fractions >= border_min_fraction) & ~core
    tags = np.full(fractions.shape, healthy_tag, dtype=np.int32)
    tags[border] = border_zone_tag
    tags[core] = scar_core_tag
    unique, counts = np.unique(tags, return_counts=True)
    return tags, {
        "healthy_tag": int(healthy_tag),
        "scar_core_tag": int(scar_core_tag),
        "border_zone_tag": int(border_zone_tag),
        "border_min_fraction": float(border_min_fraction),
        "core_min_fraction": float(core_min_fraction),
        "centroid_inside_promoted_to_core": bool(promote_centroid_to_core),
        "healthy_count": int(np.count_nonzero(tags == healthy_tag)),
        "scar_core_count": int(np.count_nonzero(tags == scar_core_tag)),
        "border_zone_count": int(np.count_nonzero(tags == border_zone_tag)),
        "scar_or_border_count": int(np.count_nonzero(tags != healthy_tag)),
        "tag_counts": {str(int(key)): int(value) for key, value in zip(unique, counts)},
    }


def write_multitag_elem(src_elem: Path, dst_elem: Path, tags: np.ndarray) -> None:
    dst_elem.parent.mkdir(parents=True, exist_ok=True)
    element_index = 0
    with (
        src_elem.open("r", encoding="utf-8", errors="replace") as source,
        dst_elem.open("w", encoding="utf-8") as target,
    ):
        header = source.readline()
        if not header:
            raise ValueError(f"Empty elem file: {src_elem}")
        target.write(header)
        for line in source:
            fields = line.split()
            if not fields:
                target.write(line)
                continue
            if fields[0] == "Tt":
                if element_index >= tags.size:
                    raise ValueError("Source elem contains more tetrahedra than tags")
                fields[-1] = str(int(tags[element_index]))
                target.write(" ".join(fields) + "\n")
                element_index += 1
            else:
                target.write(line)
    if element_index != tags.size:
        raise ValueError(f"Tagged {element_index} tetrahedra, expected {tags.size}")


def write_fraction_vtu(
    path: Path,
    points: np.ndarray,
    tets: np.ndarray,
    tags: np.ndarray,
    original_tags: np.ndarray,
    fractions: np.ndarray,
    centroid_inside: np.ndarray,
    scar_core_tag: int,
    border_zone_tag: int,
) -> None:
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points.astype(np.float64, copy=False), deep=True))
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    for tetrahedron in tets:
        cell = vtk.vtkTetra()
        for local_id, point_id in enumerate(tetrahedron):
            cell.GetPointIds().SetId(local_id, int(point_id))
        grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

    node_scar = np.zeros(points.shape[0], dtype=np.uint8)
    touched = fractions > 0
    if np.any(touched):
        node_scar[np.unique(tets[touched].ravel())] = 1
    arrays = (
        ("region_tag", tags.astype(np.int32), vtk.VTK_INT, "cell"),
        ("original_region_tag", original_tags.astype(np.int32), vtk.VTK_INT, "cell"),
        ("raw_scar_fraction", fractions.astype(np.float32), vtk.VTK_FLOAT, "cell"),
        ("raw_scar_core", (tags == scar_core_tag).astype(np.uint8), vtk.VTK_UNSIGNED_CHAR, "cell"),
        ("raw_scar_border_zone", (tags == border_zone_tag).astype(np.uint8), vtk.VTK_UNSIGNED_CHAR, "cell"),
        ("raw_scar_centroid_inside", centroid_inside.astype(np.uint8), vtk.VTK_UNSIGNED_CHAR, "cell"),
        ("raw_scar_touched_node", node_scar, vtk.VTK_UNSIGNED_CHAR, "point"),
    )
    for name, data, vtk_type, association in arrays:
        array = numpy_to_vtk(data, deep=True, array_type=vtk_type)
        array.SetName(name)
        (grid.GetCellData() if association == "cell" else grid.GetPointData()).AddArray(array)
    grid.GetCellData().SetActiveScalars("raw_scar_fraction")

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetDataModeToBinary()
    if writer.Write() != 1 or not path.is_file():
        raise OSError(f"Failed to write fraction VTU: {path}")


def surface_qc(poly: vtk.vtkPolyData) -> dict[str, Any]:
    return {
        "points": int(poly.GetNumberOfPoints()),
        "cells": int(poly.GetNumberOfCells()),
        "boundary_edges": count_feature_edges(poly, boundary=True, non_manifold=False),
        "non_manifold_edges": count_feature_edges(poly, boundary=False, non_manifold=True),
    }


def maybe_write_mask_surface(
    mask: np.ndarray,
    path: Path,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> tuple[str | None, dict[str, Any] | None]:
    if not mask.any():
        return None, None
    poly = mask_to_surface(mask, spacing_zyx, origin_xyz, 20, 0.08)
    write_surface(path, poly)
    return str(path), surface_qc(poly)


def process_case(
    case: str,
    root: Path,
    config_path: Path,
    output_root: Path,
    version: str,
    scar_label_min: float,
    scar_inner_mm: float,
    scar_outer_mm: float,
    geometry_margin_mm: float,
    bridge_mm: float,
    sdf_sigma_mm: float,
    smooth_iterations: int,
    pass_band: float,
    crop_padding_mm: float,
    scale: float,
    quality: int,
    preserve_boundary: int,
    sample_order: int,
    sample_chunk_size: int,
    border_min_fraction: float,
    core_min_fraction: float,
    healthy_tag: int,
    scar_core_tag: int,
    border_zone_tag: int,
    promote_centroid_to_core: bool,
    make_default_fibers: bool,
    default_fiber_bath_tags: str,
    default_fiber_op: int,
) -> dict[str, Any]:
    if version != "closed":
        raise ValueError("The scar-aware trial currently supports only closed geometry")

    config = load_json(config_path)
    case_dir = output_root / case / version
    surface_dir = case_dir / "surfaces"
    case_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{case}_{version}_scaraware"

    atrium_path = root / case / "atriumSegImgMO.nii.gz"
    atrium, source_meta = read_nifti(atrium_path)
    params = resolve_geometry_params(config, atrium, source_meta)
    lumen, mask_source = load_lumen_mask(atrium, params)
    lumen = clean_lumen(lumen, source_meta.spacing_zyx, source_meta.voxel_volume_mm3, params)
    target_spacing_mm = float(params["geometry_workspace_spacing_mm"])
    lumen_iso, spacing_zyx, zoom_factors = resample_to_isotropic(
        lumen, source_meta.spacing_zyx, target_spacing_mm
    )
    voxel_volume = float(np.prod(spacing_zyx))
    lumen_iso = clean_lumen(lumen_iso, spacing_zyx, voxel_volume, params)

    raw_scar_iso, raw_scar_origin_xyz, raw_scar_info = load_raw_scar_iso(
        case, root, target_spacing_mm, scar_label_min
    )
    if raw_scar_iso.shape != lumen_iso.shape:
        raise ValueError(f"Scar shape {raw_scar_iso.shape} does not match lumen shape {lumen_iso.shape}")
    if not np.allclose(raw_scar_origin_xyz, source_meta.origin_xyz, atol=1e-4):
        raise ValueError(
            f"Scar origin {raw_scar_origin_xyz} does not match atrium origin {source_meta.origin_xyz}"
        )

    masks, geometry_info = build_scar_aware_masks(
        lumen_iso,
        raw_scar_iso,
        spacing_zyx,
        float(params["wall_out_mm"]),
        scar_inner_mm,
        scar_outer_mm,
        geometry_margin_mm,
        bridge_mm,
    )
    iso_meta = make_isotropic_meta(source_meta, masks["wall"], target_spacing_mm)
    mask_outputs = {
        "scar_aware_wall_mask": str(case_dir / f"{prefix}_wall_mask.nii.gz"),
        "scar_aware_lumen_mask": str(case_dir / f"{prefix}_lumen_mask.nii.gz"),
        "scar_aware_outer_solid_mask": str(case_dir / f"{prefix}_outer_solid_mask.nii.gz"),
        "raw_scar_isotropic_mask": str(case_dir / f"{prefix}_raw_scar_isotropic_mask.nii.gz"),
        "plausible_raw_scar_mask": str(case_dir / f"{prefix}_plausible_raw_scar_mask.nii.gz"),
        "raw_scar_inside_scar_aware_wall_mask": str(case_dir / f"{prefix}_raw_scar_inside_wall_mask.nii.gz"),
        "geometry_support_mask": str(case_dir / f"{prefix}_geometry_support_mask.nii.gz"),
    }
    for key, mask in (
        ("scar_aware_wall_mask", masks["wall"]),
        ("scar_aware_lumen_mask", masks["lumen"]),
        ("scar_aware_outer_solid_mask", masks["outer"]),
        ("raw_scar_isotropic_mask", masks["raw_scar"]),
        ("plausible_raw_scar_mask", masks["plausible_scar"]),
        ("raw_scar_inside_scar_aware_wall_mask", masks["raw_scar"] & masks["wall"]),
        ("geometry_support_mask", masks["support"]),
    ):
        write_nifti(Path(mask_outputs[key]), mask, iso_meta)

    surfaces = build_boundary_surfaces(
        masks["lumen"], masks["outer"], spacing_zyx, source_meta.origin_xyz,
        surface_dir, prefix, sdf_sigma_mm, smooth_iterations, pass_band, crop_padding_mm,
    )
    raw_surface_path, raw_surface_qc = maybe_write_mask_surface(
        masks["raw_scar"], surface_dir / f"{prefix}_raw_scar_show3d_surface.vtp",
        spacing_zyx, source_meta.origin_xyz,
    )
    plausible_surface_path, plausible_surface_qc = maybe_write_mask_surface(
        masks["plausible_scar"], surface_dir / f"{prefix}_plausible_raw_scar_surface.vtp",
        spacing_zyx, source_meta.origin_xyz,
    )

    mesh_base = case_dir / f"{prefix}_tet"
    generation = run_meshtool_generate(
        Path(surfaces["endocardium"]["obj"]), Path(surfaces["epicardium"]["obj"]),
        mesh_base, scale, quality, preserve_boundary,
    )
    report_path = case_dir / f"{prefix}_qc.json"
    report: dict[str, Any] = {
        "case": case,
        "version": version,
        "config": str(config_path),
        "effective_params": params,
        "mask_source": mask_source,
        "source_shape_zyx": [int(value) for value in atrium.shape],
        "source_spacing_zyx_mm": [float(value) for value in source_meta.spacing_zyx],
        "iso_shape_zyx": [int(value) for value in lumen_iso.shape],
        "iso_spacing_zyx_mm": [float(value) for value in spacing_zyx],
        "zoom_factors_zyx": [float(value) for value in zoom_factors],
        "raw_scar_source": raw_scar_info,
        "scar_aware_geometry": geometry_info,
        "parameters": {
            "sdf_sigma_mm": float(sdf_sigma_mm),
            "smooth_iterations": int(smooth_iterations),
            "pass_band": float(pass_band),
            "crop_padding_mm": float(crop_padding_mm),
            "meshtool_scale": float(scale),
            "meshtool_quality": int(quality),
            "meshtool_preserve_boundary": int(preserve_boundary),
            "sample_order": int(sample_order),
            "sample_chunk_size": int(sample_chunk_size),
            "border_min_fraction": float(border_min_fraction),
            "core_min_fraction": float(core_min_fraction),
            "promote_centroid_to_core": bool(promote_centroid_to_core),
        },
        "outputs": {
            "masks": mask_outputs,
            "surfaces": {
                "boundary_surfaces": surfaces,
                "raw_scar_show3d_surface": raw_surface_path,
                "raw_scar_show3d_surface_qc": raw_surface_qc,
                "plausible_raw_scar_surface": plausible_surface_path,
                "plausible_raw_scar_surface_qc": plausible_surface_qc,
            },
            "mesh_base": str(mesh_base),
            "pts": str(mesh_base.with_suffix(".pts")),
            "elem": str(mesh_base.with_suffix(".elem")),
            "lon": str(mesh_base.with_suffix(".lon")),
        },
        "meshtool_generate": generation,
        "tetra_qc": None,
        "scar_fraction_mapping": None,
        "meshtool_quality": None,
        "default_fibers": None,
        "report": str(report_path),
    }

    if generation["status"] == "generated":
        points = read_pts(mesh_base.with_suffix(".pts"))
        tets, original_tags, type_counts = read_elem(mesh_base.with_suffix(".elem"))
        fractions, centroid_inside, sampling_info = sample_scar_fraction(
            points, tets, masks["raw_scar"], spacing_zyx, source_meta.origin_xyz,
            sample_order, sample_chunk_size,
        )
        tags, tag_info = classify_fraction_tags(
            fractions, centroid_inside, border_min_fraction, core_min_fraction,
            healthy_tag, scar_core_tag, border_zone_tag, promote_centroid_to_core,
        )
        tagged_base = case_dir / f"{prefix}_rawfraction_tet"
        shutil.copyfile(mesh_base.with_suffix(".pts"), tagged_base.with_suffix(".pts"))
        if mesh_base.with_suffix(".lon").is_file():
            shutil.copyfile(mesh_base.with_suffix(".lon"), tagged_base.with_suffix(".lon"))
        write_multitag_elem(mesh_base.with_suffix(".elem"), tagged_base.with_suffix(".elem"), tags)
        tagged_vtu = tagged_base.with_suffix(".vtu")
        write_fraction_vtu(
            tagged_vtu, points, tets, tags, original_tags, fractions, centroid_inside,
            scar_core_tag, border_zone_tag,
        )
        fraction_path = case_dir / f"{prefix}_raw_scar_fraction.dat"
        tag_path = case_dir / f"{prefix}_region_tag.dat"
        write_scalar_dat(fraction_path, fractions)
        write_scalar_dat(tag_path, tags)

        volumes = tetra_volumes(points, tets)
        unique_tags, tag_counts = np.unique(tags, return_counts=True)
        tag_volume_ml = {
            str(int(tag)): float(np.sum(volumes[tags == tag]) / 1000.0)
            for tag in unique_tags
        }
        raw_source_ml = float(raw_scar_info["source_ml"])
        fraction_integral_volume_ml = float(np.sum(fractions.astype(np.float64) * volumes) / 1000.0)
        core_volume_ml = float(np.sum(volumes[tags == scar_core_tag]) / 1000.0)
        border_volume_ml = float(np.sum(volumes[tags == border_zone_tag]) / 1000.0)
        touched_volume_ml = float(np.sum(volumes[fractions > 0]) / 1000.0)

        report["outputs"].update({
            "tagged_mesh_base": str(tagged_base),
            "tagged_pts": str(tagged_base.with_suffix(".pts")),
            "tagged_elem": str(tagged_base.with_suffix(".elem")),
            "tagged_lon": str(tagged_base.with_suffix(".lon")) if tagged_base.with_suffix(".lon").is_file() else None,
            "tagged_vtu": str(tagged_vtu),
            "raw_scar_fraction_dat": str(fraction_path),
            "region_tag_dat": str(tag_path),
        })
        report["tetra_qc"] = {
            "element_type_counts": type_counts,
            **tetra_quality(points, tets, original_tags),
        }
        wall_ml = float(mask_stats(masks["wall"], spacing_zyx)["ml"])
        mesh_ml = float(report["tetra_qc"]["total_abs_volume_ml"])
        report["volume_check"] = {
            "scar_aware_wall_mask_volume_ml": wall_ml,
            "tetra_mesh_abs_volume_ml": mesh_ml,
            "relative_error": abs(mesh_ml - wall_ml) / wall_ml if wall_ml > 0 else None,
        }
        report["scar_fraction_mapping"] = {
            **sampling_info,
            **tag_info,
            "tag_counts": {str(int(tag)): int(count) for tag, count in zip(unique_tags, tag_counts)},
            "tag_volume_ml": tag_volume_ml,
            "raw_source_scar_ml": raw_source_ml,
            "fraction_integral_scar_volume_ml": fraction_integral_volume_ml,
            "fraction_integral_to_raw_source_volume_ratio": fraction_integral_volume_ml / raw_source_ml if raw_source_ml > 0 else None,
            "scar_core_volume_ml": core_volume_ml,
            "border_zone_volume_ml": border_volume_ml,
            "core_plus_border_volume_ml": core_volume_ml + border_volume_ml,
            "all_touched_scar_fraction_gt0_volume_ml": touched_volume_ml,
            "note": "Border zone is direct sampled partial overlap, not morphology expansion.",
        }
        report["meshtool_quality"] = run_meshtool_quality(mesh_base)
        if make_default_fibers:
            report["default_fibers"] = generate_default_fibers(
                tagged_base, case_dir / f"{prefix}_rawfraction_defaultfibers_tet",
                default_fiber_bath_tags, default_fiber_op,
            )
        report["recommendation"] = {
            "ready_for_paraview_review": True,
            "open_in_paraview_first": str(tagged_vtu),
            "color_by": "raw_scar_fraction first, then region_tag",
            "uses_raw_expert_scar_without_projection_or_dilation": True,
            "caveat": "Experimental scar-aware geometry; review before EP parameter trials.",
        }
    else:
        report["recommendation"] = {
            "ready_for_paraview_review": False,
            "next_action": "Inspect endocardium and epicardium surfaces before retrying meshtool.",
        }

    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="train_1")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("step05_outputs/scar_aware_pipeline"))
    parser.add_argument("--version", choices=("closed",), default="closed")
    parser.add_argument("--scar-label-min", type=float, default=0.5)
    parser.add_argument("--scar-inner-mm", type=float, default=4.0)
    parser.add_argument("--scar-outer-mm", type=float, default=6.0)
    parser.add_argument("--geometry-margin-mm", type=float, default=0.625)
    parser.add_argument("--bridge-mm", type=float, default=1.25)
    parser.add_argument("--sdf-sigma-mm", type=float, default=0.9)
    parser.add_argument("--smooth-iterations", type=int, default=80)
    parser.add_argument("--pass-band", type=float, default=0.025)
    parser.add_argument("--crop-padding-mm", type=float, default=8.0)
    parser.add_argument("--scale", type=float, default=1.6)
    parser.add_argument("--quality", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--preserve-boundary", type=int, choices=(0, 1), default=1)
    parser.add_argument("--sample-order", type=int, default=4)
    parser.add_argument("--sample-chunk-size", type=int, default=40000)
    parser.add_argument("--border-min-fraction", type=float, default=0.02)
    parser.add_argument("--core-min-fraction", type=float, default=0.25)
    parser.add_argument("--healthy-tag", type=int, default=1)
    parser.add_argument("--scar-core-tag", type=int, default=2)
    parser.add_argument("--border-zone-tag", type=int, default=3)
    parser.add_argument("--promote-centroid-to-core", action="store_true")
    parser.add_argument("--skip-default-fibers", action="store_true")
    parser.add_argument("--default-fiber-bath-tags", default="0")
    parser.add_argument("--default-fiber-op", type=int, default=1)
    args = parser.parse_args()

    report = process_case(
        case=args.case,
        root=args.root.resolve(),
        config_path=resolve_config_path(args.config),
        output_root=args.output_root.resolve(),
        version=args.version,
        scar_label_min=args.scar_label_min,
        scar_inner_mm=args.scar_inner_mm,
        scar_outer_mm=args.scar_outer_mm,
        geometry_margin_mm=args.geometry_margin_mm,
        bridge_mm=args.bridge_mm,
        sdf_sigma_mm=args.sdf_sigma_mm,
        smooth_iterations=args.smooth_iterations,
        pass_band=args.pass_band,
        crop_padding_mm=args.crop_padding_mm,
        scale=args.scale,
        quality=args.quality,
        preserve_boundary=args.preserve_boundary,
        sample_order=args.sample_order,
        sample_chunk_size=args.sample_chunk_size,
        border_min_fraction=args.border_min_fraction,
        core_min_fraction=args.core_min_fraction,
        healthy_tag=args.healthy_tag,
        scar_core_tag=args.scar_core_tag,
        border_zone_tag=args.border_zone_tag,
        promote_centroid_to_core=args.promote_centroid_to_core,
        make_default_fibers=not args.skip_default_fibers,
        default_fiber_bath_tags=args.default_fiber_bath_tags,
        default_fiber_op=args.default_fiber_op,
    )
    print(json.dumps({
        "case": report["case"],
        "report": report["report"],
        "meshtool_generate_status": report["meshtool_generate"]["status"],
        "outputs": report["outputs"],
        "scar_aware_geometry": report["scar_aware_geometry"],
        "scar_fraction_mapping": report["scar_fraction_mapping"],
        "recommendation": report.get("recommendation"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
