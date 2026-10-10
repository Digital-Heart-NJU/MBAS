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
    sampled = ndi.map_coordinates(
        mask.astype(np.float32), coordinates, order=0, mode="constant", cval=0.0
    )
    return sampled >= 0.5

def tetra_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    tetra_points = points[tets]
    signed = np.einsum(
        "ij,ij->i",
        np.cross(tetra_points[:, 1] - tetra_points[:, 0], tetra_points[:, 2] - tetra_points[:, 0]),
        tetra_points[:, 3] - tetra_points[:, 0],
    ) / 6.0
    return np.abs(signed)


def classify_tetra_by_raw_overlap(
    points: np.ndarray,
    tets: np.ndarray,
    scar_iso: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    min_sample_fraction: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not 0 <= min_sample_fraction <= 1:
        raise ValueError("min_sample_fraction must be between 0 and 1")
    if chunk_size <= 0:
        raise ValueError("sample_chunk_size must be positive")

    sample_fraction = np.zeros(tets.shape[0], dtype=np.float32)
    centroid_inside = np.zeros(tets.shape[0], dtype=bool)
    volumes = tetra_volumes(points, tets)
    for start in range(0, tets.shape[0], chunk_size):
        end = min(start + chunk_size, tets.shape[0])
        samples = tetra_sample_points(points, tets[start:end])
        inside = sample_binary_mask_at_points(
            scar_iso,
            samples.reshape((-1, 3)),
            spacing_zyx,
            origin_xyz,
        ).reshape(samples.shape[:2])
        centroid_inside[start:end] = inside[:, 0]
        sample_fraction[start:end] = inside.mean(axis=1)

    scar_elements = (sample_fraction >= min_sample_fraction) | centroid_inside
    scar_volume_ml = float(np.sum(volumes[scar_elements]) / 1000.0)
    total_volume_ml = float(np.sum(volumes) / 1000.0)
    return scar_elements, sample_fraction, volumes, {
        "method": "direct raw scar volume overlap sampling",
        "sample_points_per_tetra": 11,
        "sample_points": "centroid, four vertices, six edge midpoints",
        "min_sample_fraction": float(min_sample_fraction),
        "centroid_inside_always_counts_as_scar": True,
        "tetra_count": int(tets.shape[0]),
        "scar_element_count": int(np.count_nonzero(scar_elements)),
        "scar_element_fraction": float(np.mean(scar_elements)) if scar_elements.size else None,
        "scar_tetra_volume_ml": scar_volume_ml,
        "total_tetra_volume_ml": total_volume_ml,
        "scar_tetra_volume_fraction": scar_volume_ml / total_volume_ml if total_volume_ml > 0 else None,
        "raw_scar_sample_fraction": percentile_stats(sample_fraction),
    }

def scar_wall_consistency(
    scar_iso: np.ndarray,
    lumen: np.ndarray,
    wall: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> dict[str, Any]:
    if scar_iso.shape != lumen.shape or scar_iso.shape != wall.shape:
        raise ValueError("Scar, lumen, and wall masks must have identical shapes")
    scar_count = int(scar_iso.sum())
    if scar_count == 0:
        return {"scar_voxels": 0, "scar_ml": 0.0}

    outside_dist = ndi.distance_transform_edt(~lumen, sampling=spacing_zyx)
    inside_dist = ndi.distance_transform_edt(lumen, sampling=spacing_zyx)
    signed = outside_dist.astype(np.float32, copy=False)
    signed[lumen] = -inside_dist[lumen]
    in_wall = scar_iso & wall
    in_lumen = scar_iso & lumen
    outside_lumen_wall = scar_iso & ~(lumen | wall)
    return {
        "scar_voxels": scar_count,
        "scar_ml": float(scar_count * np.prod(spacing_zyx) / 1000.0),
        "raw_scar_inside_generated_wall_fraction": float(in_wall.sum() / scar_count),
        "raw_scar_inside_lumen_fraction": float(in_lumen.sum() / scar_count),
        "raw_scar_outside_lumen_and_wall_fraction": float(outside_lumen_wall.sum() / scar_count),
        "signed_distance_to_lumen_boundary_for_raw_scar_mm": percentile_stats(
            signed[scar_iso], "mm"
        ),
        "note": "Fractions describe registration/wall mismatch; no scar voxels are moved or expanded.",
    }

def write_tagged_elem(
    src_elem: Path,
    dst_elem: Path,
    scar_elements: np.ndarray,
    healthy_tag: int,
    scar_tag: int,
) -> None:
    dst_elem.parent.mkdir(parents=True, exist_ok=True)
    tetra_index = 0
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
                if tetra_index >= scar_elements.size:
                    raise ValueError("Source elem contains more tetrahedra than classification")
                fields[-1] = str(scar_tag if scar_elements[tetra_index] else healthy_tag)
                target.write(" ".join(fields) + "\n")
                tetra_index += 1
            else:
                target.write(line)
    if tetra_index != scar_elements.size:
        raise ValueError(f"Tagged {tetra_index} tetrahedra, expected {scar_elements.size}")


def write_scalar_dat(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, fmt="%g")


def make_vtu_grid(
    points: np.ndarray,
    tets: np.ndarray,
    scar_elements: np.ndarray,
    original_tags: np.ndarray,
    sample_fraction: np.ndarray,
    healthy_tag: int,
    scar_tag: int,
) -> vtk.vtkUnstructuredGrid:
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
    if np.any(scar_elements):
        node_scar[np.unique(tets[scar_elements].ravel())] = 1
    arrays = (
        ("region_tag", np.where(scar_elements, scar_tag, healthy_tag).astype(np.int32), vtk.VTK_INT, "cell"),
        ("original_region_tag", original_tags.astype(np.int32), vtk.VTK_INT, "cell"),
        ("raw_scar_element", scar_elements.astype(np.uint8), vtk.VTK_UNSIGNED_CHAR, "cell"),
        ("raw_scar_sample_fraction", sample_fraction.astype(np.float32), vtk.VTK_FLOAT, "cell"),
        ("raw_scar_node", node_scar, vtk.VTK_UNSIGNED_CHAR, "point"),
    )
    for name, data, vtk_type, association in arrays:
        array = numpy_to_vtk(data, deep=True, array_type=vtk_type)
        array.SetName(name)
        if association == "cell":
            grid.GetCellData().AddArray(array)
        else:
            grid.GetPointData().AddArray(array)
    grid.GetCellData().SetActiveScalars("region_tag")
    return grid


def write_vtu(path: Path, grid: vtk.vtkUnstructuredGrid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetDataModeToBinary()
    if writer.Write() != 1 or not path.is_file():
        raise OSError(f"Failed to write VTU file: {path}")


def extract_scar_tetra_surface(grid: vtk.vtkUnstructuredGrid) -> vtk.vtkPolyData:
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(grid)
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "raw_scar_element"
    )
    threshold.SetLowerThreshold(1.0)
    threshold.SetUpperThreshold(1.0)
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    threshold.Update()

    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputConnection(threshold.GetOutputPort())
    surface.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(surface.GetOutputPort())
    tri.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def run_command(cmd: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        )
    except OSError as exc:
        output = f"{type(exc).__name__}: {exc}\n"
        log_path.write_text(output, encoding="utf-8")
        return {"cmd": cmd, "returncode": None, "log": str(log_path), "stdout_tail": output, "error": str(exc)}
    log_path.write_text(result.stdout, encoding="utf-8")
    return {
        "cmd": cmd,
        "returncode": int(result.returncode),
        "log": str(log_path),
        "stdout_tail": result.stdout[-3000:],
    }


def find_meshtool() -> str | None:
    found = shutil.which("meshtool")
    if found:
        return found
    fallback = Path("/usr/local/lib/opencarp/bin/meshtool")
    return str(fallback) if fallback.is_file() else None


def generate_default_fibers(
    input_base: Path,
    output_base: Path,
    bath_tags: str,
    op: int,
) -> dict[str, Any]:
    meshtool = find_meshtool()
    if meshtool is None:
        return {"status": "unavailable", "reason": "meshtool executable was not found"}
    result = run_command(
        [
            meshtool,
            "generate", "fibres",
            f"-msh={input_base}",
            f"-tags={bath_tags}",
            f"-outmsh={output_base}",
            f"-op={op}",
        ],
        output_base.parent / f"{output_base.name}_generate_fibres.log",
    )
    if result.get("returncode") == 0 and output_base.with_suffix(".lon").is_file():
        shutil.copyfile(input_base.with_suffix(".pts"), output_base.with_suffix(".pts"))
        shutil.copyfile(input_base.with_suffix(".elem"), output_base.with_suffix(".elem"))
        result["status"] = "generated"
        result["outputs"] = {
            "mesh_base": str(output_base),
            "pts": str(output_base.with_suffix(".pts")),
            "elem": str(output_base.with_suffix(".elem")),
            "lon": str(output_base.with_suffix(".lon")),
        }
    else:
        result["status"] = "failed"
    return result


def process_case(
    case: str,
    root: Path,
    config_path: Path,
    tetra_root: Path,
    output_root: Path,
    version: str,
    scar_label_min: float,
    min_sample_fraction: float,
    sample_chunk_size: int,
    surface_smooth_iterations: int,
    surface_pass_band: float,
    healthy_tag: int,
    scar_tag: int,
    make_default_fibers: bool,
    default_fiber_bath_tags: str,
    default_fiber_op: int,
) -> dict[str, Any]:
    config = load_json(config_path)
    masks, spacing_zyx, origin_xyz, iso_meta, wall_info = reconstruct_wall_masks(
        case, root, config, version
    )
    scar_iso, scar_origin_xyz, scar_info = load_raw_scar_iso(
        case, root, spacing_zyx[0], scar_label_min
    )
    if scar_iso.shape != masks["wall"].shape:
        raise ValueError(
            f"Scar shape {scar_iso.shape} does not match geometry shape {masks['wall'].shape}; "
            "check the NIfTI registration and spacing before mapping."
        )
    if not np.allclose(scar_origin_xyz, origin_xyz, atol=1e-4):
        raise ValueError(
            f"Scar origin {scar_origin_xyz} does not match atrium origin {origin_xyz}; "
            "strict mapping refuses an unregistered mask."
        )

    mesh_base = tetra_root / case / version / f"{case}_{version}_tet"
    if not mesh_base.with_suffix(".pts").is_file() or not mesh_base.with_suffix(".elem").is_file():
        raise FileNotFoundError(
            f"Tetra mesh is incomplete; expected {mesh_base.with_suffix('.pts')} and "
            f"{mesh_base.with_suffix('.elem')}"
        )
    points = read_pts(mesh_base.with_suffix(".pts"))
    tets, original_tags, type_counts = read_elem(mesh_base.with_suffix(".elem"))
    scar_elements, sample_fraction, _, element_info = classify_tetra_by_raw_overlap(
        points,
        tets,
        scar_iso,
        spacing_zyx,
        origin_xyz,
        min_sample_fraction,
        sample_chunk_size,
    )

    out_dir = output_root / case / version
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{case}_{version}_rawscar"
    tagged_base = out_dir / f"{prefix}_tet"
    shutil.copyfile(mesh_base.with_suffix(".pts"), tagged_base.with_suffix(".pts"))
    if mesh_base.with_suffix(".lon").is_file():
        shutil.copyfile(mesh_base.with_suffix(".lon"), tagged_base.with_suffix(".lon"))
    write_tagged_elem(
        mesh_base.with_suffix(".elem"),
        tagged_base.with_suffix(".elem"),
        scar_elements,
        healthy_tag,
        scar_tag,
    )

    grid = make_vtu_grid(
        points, tets, scar_elements, original_tags, sample_fraction, healthy_tag, scar_tag
    )
    vtu_path = tagged_base.with_suffix(".vtu")
    write_vtu(vtu_path, grid)

    raw_scar_mask_path = out_dir / f"{prefix}_isotropic_mask.nii.gz"
    raw_scar_in_wall_path = out_dir / f"{prefix}_inside_generated_wall_mask.nii.gz"
    write_nifti(raw_scar_mask_path, scar_iso, iso_meta)
    write_nifti(raw_scar_in_wall_path, scar_iso & masks["wall"], iso_meta)

    raw_surface = mask_to_surface(
        scar_iso, spacing_zyx, origin_xyz, surface_smooth_iterations, surface_pass_band
    )
    raw_surface_path = out_dir / f"{prefix}_show3d_surface.vtp"
    write_surface(raw_surface_path, raw_surface)

    scar_tetra_surface = extract_scar_tetra_surface(grid)
    scar_tetra_surface_path = out_dir / f"{prefix}_tetra_label_surface.vtp"
    if scar_tetra_surface.GetNumberOfCells() > 0:
        write_surface(scar_tetra_surface_path, scar_tetra_surface)

    binary_path = out_dir / f"{prefix}_element_binary.dat"
    fraction_path = out_dir / f"{prefix}_element_sample_fraction.dat"
    write_scalar_dat(binary_path, scar_elements.astype(np.uint8))
    write_scalar_dat(fraction_path, sample_fraction)

    default_fibers = None
    if make_default_fibers:
        default_fibers = generate_default_fibers(
            tagged_base,
            out_dir / f"{prefix}_defaultfibers_tet",
            default_fiber_bath_tags,
            default_fiber_op,
        )

    raw_source_ml = float(scar_info["source_ml"])
    mapped_volume_ml = float(element_info["scar_tetra_volume_ml"])
    report_path = out_dir / f"{prefix}_tetra_mapping_qc.json"
    report: dict[str, Any] = {
        "case": case,
        "version": version,
        "config": str(config_path),
        "source_tetra_mesh_base": str(mesh_base),
        "wall_reconstruction": wall_info,
        "raw_scar_source": scar_info,
        "raw_scar_vs_generated_wall": scar_wall_consistency(
            scar_iso, masks["lumen"], masks["wall"], spacing_zyx
        ),
        "tetra_input": {
            "point_count": int(points.shape[0]),
            "element_type_counts": type_counts,
            "original_tag_counts": {
                str(int(tag)): int(count)
                for tag, count in zip(*np.unique(original_tags, return_counts=True))
            },
        },
        "element_classification": element_info,
        "volume_comparison": {
            "raw_source_scar_ml": raw_source_ml,
            "mapped_scar_tetra_ml": mapped_volume_ml,
            "mapped_to_raw_source_volume_ratio": mapped_volume_ml / raw_source_ml if raw_source_ml > 0 else None,
            "note": "The ratio need not be 1.0 because voxel labels are converted to tetrahedral cell labels without expansion.",
        },
        "visual_surfaces": {
            "raw_expert_scar_show3d_surface": str(raw_surface_path),
            "raw_expert_scar_show3d_surface_qc": {
                "points": int(raw_surface.GetNumberOfPoints()),
                "cells": int(raw_surface.GetNumberOfCells()),
                "boundary_edges": count_feature_edges(raw_surface, boundary=True, non_manifold=False),
                "non_manifold_edges": count_feature_edges(raw_surface, boundary=False, non_manifold=True),
            },
            "scar_tetra_label_surface": str(scar_tetra_surface_path) if scar_tetra_surface.GetNumberOfCells() else None,
            "scar_tetra_label_surface_qc": {
                "points": int(scar_tetra_surface.GetNumberOfPoints()),
                "cells": int(scar_tetra_surface.GetNumberOfCells()),
                "boundary_edges": count_feature_edges(scar_tetra_surface, boundary=True, non_manifold=False) if scar_tetra_surface.GetNumberOfCells() else 0,
                "non_manifold_edges": count_feature_edges(scar_tetra_surface, boundary=False, non_manifold=True) if scar_tetra_surface.GetNumberOfCells() else 0,
            },
        },
        "outputs": {
            "tagged_mesh_base": str(tagged_base),
            "pts": str(tagged_base.with_suffix(".pts")),
            "elem": str(tagged_base.with_suffix(".elem")),
            "lon": str(tagged_base.with_suffix(".lon")) if tagged_base.with_suffix(".lon").is_file() else None,
            "vtu": str(vtu_path),
            "raw_scar_isotropic_mask": str(raw_scar_mask_path),
            "raw_scar_inside_generated_wall_mask": str(raw_scar_in_wall_path),
            "element_binary_dat": str(binary_path),
            "element_sample_fraction_dat": str(fraction_path),
        },
        "default_fibers": default_fibers,
        "tag_legend": {
            str(healthy_tag): "generated LA myocardium without raw expert scar overlap",
            str(scar_tag): "raw expert scar overlap on tetrahedral elements",
        },
        "recommendation": {
            "ready_for_paraview_review": bool(np.count_nonzero(scar_elements) > 0),
            "uses_raw_expert_scar_without_projection_or_expansion": True,
            "ready_for_solver_smoke_test_with_default_fibers": bool(
                np.count_nonzero(scar_elements) > 0
                and default_fibers is not None
                and default_fibers.get("status") == "generated"
            ),
            "ready_for_final_ep_parameter_trial": False,
            "next_review": "Inspect the raw scar surface and scar-labelled VTU in ParaView before solver use.",
        },
        "report": str(report_path),
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="train_1")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tetra-root", type=Path, default=Path("step03_outputs/tetra_mesh"))
    parser.add_argument("--output-root", type=Path, default=Path("step04_outputs/raw_scar_on_tetra"))
    parser.add_argument("--version", choices=("closed", "open_candidate"), default="closed")
    parser.add_argument("--scar-label-min", type=float, default=0.5)
    parser.add_argument("--min-sample-fraction", type=float, default=0.10)
    parser.add_argument("--sample-chunk-size", type=int, default=75000)
    parser.add_argument("--surface-smooth-iterations", type=int, default=20)
    parser.add_argument("--surface-pass-band", type=float, default=0.08)
    parser.add_argument("--healthy-tag", type=int, default=1)
    parser.add_argument("--scar-tag", type=int, default=2)
    parser.add_argument("--skip-default-fibers", action="store_true")
    parser.add_argument("--default-fiber-bath-tags", default="0")
    parser.add_argument("--default-fiber-op", type=int, default=1)
    args = parser.parse_args()

    report = process_case(
        case=args.case,
        root=args.root.resolve(),
        config_path=resolve_config_path(args.config),
        tetra_root=args.tetra_root.resolve(),
        output_root=args.output_root.resolve(),
        version=args.version,
        scar_label_min=args.scar_label_min,
        min_sample_fraction=args.min_sample_fraction,
        sample_chunk_size=args.sample_chunk_size,
        surface_smooth_iterations=args.surface_smooth_iterations,
        surface_pass_band=args.surface_pass_band,
        healthy_tag=args.healthy_tag,
        scar_tag=args.scar_tag,
        make_default_fibers=not args.skip_default_fibers,
        default_fiber_bath_tags=args.default_fiber_bath_tags,
        default_fiber_op=args.default_fiber_op,
    )
    print(json.dumps({
        "case": report["case"],
        "version": report["version"],
        "report": report["report"],
        "outputs": report["outputs"],
        "raw_scar_vs_generated_wall": report["raw_scar_vs_generated_wall"],
        "element_classification": report["element_classification"],
        "volume_comparison": report["volume_comparison"],
        "recommendation": report["recommendation"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()

