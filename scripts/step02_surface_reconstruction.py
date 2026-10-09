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









