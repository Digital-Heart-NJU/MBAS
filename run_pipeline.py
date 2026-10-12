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

def load_opening_landmarks(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load reviewed, case-local opening landmarks in relative LA coordinates."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Opening landmark file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise ValueError(f"{path}: expected a top-level 'cases' object")
    for case, spec in cases.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("openings"), list):
            raise ValueError(f"{path}: case {case!r} must contain an 'openings' list")
        classes: set[str] = set()
        for opening in spec["openings"]:
            opening_class = str(opening.get("opening_class", ""))
            target = opening.get("relative_centroid_zyx")
            if not opening_class or opening_class in classes:
                raise ValueError(f"{path}: case {case!r} has a missing or duplicate opening class")
            if not isinstance(target, list) or len(target) != 3 or not all(np.isfinite(float(value)) for value in target):
                raise ValueError(f"{path}: invalid relative_centroid_zyx for {case}/{opening_class}")
            classes.add(opening_class)
    return cases


def min_contact_voxels_from_physical_threshold(
    spacing_zyx: tuple[float, float, float],
    min_contact_mm3: float,
    min_contact_voxels_floor: int,
) -> int:
    voxel_mm3 = float(math.prod(spacing_zyx))
    physical_voxels = int(math.ceil(min_contact_mm3 / voxel_mm3)) if min_contact_mm3 > 0 else 0
    return max(1, int(min_contact_voxels_floor), physical_voxels)


def opening_review_priority(contact_mm3: float, threshold_mm3: float, protected: bool) -> str:
    if not protected:
        return "none"
    if contact_mm3 < threshold_mm3 * 1.5:
        return "high"
    if contact_mm3 < threshold_mm3 * 3.0:
        return "medium"
    return "low"

def output_label_array(repaired_semantic: np.ndarray, output_mode: str) -> np.ndarray:
    if output_mode == "wall-only":
        return np.where(repaired_semantic == WALL_LABEL, WALL_LABEL, 0).astype(np.uint8)
    if output_mode == "semantic":
        return repaired_semantic.astype(np.uint8)
    raise ValueError(f"Unsupported output mode: {output_mode}")


def component_summary(mask: np.ndarray, top_n: int = 8) -> dict[str, Any]:
    cc, n_components = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(cc.ravel()) if n_components else np.array([0], dtype=np.int64)
    top = sorted((int(value) for value in sizes[1:] if value > 0), reverse=True)[:top_n]
    return {
        "component_count": int(n_components),
        "top_component_voxels": top,
        "total_voxels": int(mask.sum()),
    }


def distance_dilation(mask: np.ndarray, spacing_zyx: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    if radius_mm <= 0:
        return mask.copy()
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=bool)
    return ndi.distance_transform_edt(~mask, sampling=spacing_zyx) <= radius_mm


def distance_erosion(mask: np.ndarray, spacing_zyx: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    if radius_mm <= 0 or not np.any(mask):
        return mask.copy()
    return ndi.distance_transform_edt(mask, sampling=spacing_zyx) > radius_mm


def distance_closing(mask: np.ndarray, spacing_zyx: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    return distance_erosion(distance_dilation(mask, spacing_zyx, radius_mm), spacing_zyx, radius_mm)


def border_component_ids(cc: np.ndarray) -> set[int]:
    values = np.concatenate(
        [
            cc[0, :, :].ravel(),
            cc[-1, :, :].ravel(),
            cc[:, 0, :].ravel(),
            cc[:, -1, :].ravel(),
            cc[:, :, 0].ravel(),
            cc[:, :, -1].ravel(),
        ]
    )
    ids = {int(value) for value in np.unique(values)}
    ids.discard(0)
    return ids


