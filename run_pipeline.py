from __future__ import annotations
import argparse
import csv
import json
import math 
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import vtk
from scipy import ndimage as ndi
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy

REPRO = Path(__file__).resolve().parents[1]
DEFAULT_PRED_DIR = REPRO / "private_pred073"
DEFAULT_IMAGE_DIR = REPRO / "private_input073"
DEFAULT_OUTPUT_DIR = REPRO / "private_pred073_repaired_final"
DEFAULT_REPORT_DIR = REPRO / "private_pred073_repaired_reports"
DEFAULT_OPENING_LANDMARKS = REPRO / "private_pred073_opening_landmarks.json"

WALL_LABEL = 1
RA_LABEL = 2
LA_LABEL = 3
LABELS = {
    0: "background",
    WALL_LABEL: "biatrial_wall",
    RA_LABEL: "RA",
    LA_LABEL: "LA",
}


@dataclass(frozen=True)
class NiftiVolume:
    path: Path
    array_zyx: np.ndarray
    shape_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    qform: vtk.vtkMatrix4x4 | None
    sform: vtk.vtkMatrix4x4 | None

    @property
    def spacing_zyx(self) -> tuple[float, float, float]:
        sx, sy, sz = self.spacing_xyz
        return sz, sy, sx

def read_nifti(path: Path) -> NiftiVolume:
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = reader.GetOutput()
    dims = tuple(int(value) for value in image.GetDimensions())
    spacing = tuple(float(value) for value in image.GetSpacing())
    origin = tuple(float(value) for value in image.GetOrigin())
    scalars = image.GetPointData().GetScalars()
    if scalars is None:
        raise ValueError(f"{path}: NIfTI has no scalar data")
    array = vtk_to_numpy(scalars).reshape((dims[2], dims[1], dims[0]), order="C")
    return NiftiVolume(
        path=path,
        array_zyx=np.asarray(array),
        shape_xyz=dims,
        spacing_xyz=spacing,
        origin_xyz=origin,
        qform=reader.GetQFormMatrix(),
        sform=reader.GetSFormMatrix(),
    )


def write_nifti_like(reference: NiftiVolume, array_zyx: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vtk_array = numpy_to_vtk(array_zyx.astype(np.uint8).ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_array.SetName("label")
    image = vtk.vtkImageData()
    image.SetDimensions(*reference.shape_xyz)
    image.SetSpacing(*reference.spacing_xyz)
    image.SetOrigin(*reference.origin_xyz)
    image.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(image)
    if reference.qform is not None:
        writer.SetQFormMatrix(reference.qform)
    if reference.sform is not None:
        writer.SetSFormMatrix(reference.sform)
    writer.SetNIFTIVersion(1)
    writer.Write()

def strip_nifti_suffix(path: Path) -> str:
    for suffix in (".nii.gz", ".nii"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem

def list_niftis(directory: Path) -> list[Path]:
    return sorted([*directory.glob("*.nii.gz"), *directory.glob("*.nii")])

def matched_image_path(pred_path: Path, image_dir: Path) -> Path:
    stem = strip_nifti_suffix(pred_path)
    candidates = [
        image_dir / f"{stem}_0000.nii.gz",
        image_dir / f"{stem}_0000.nii",
        image_dir / f"{stem}.nii.gz",
        image_dir / f"{stem}.nii",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No matching image found for {pred_path.name} in {image_dir}")

def label_counts(array: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(array, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}






