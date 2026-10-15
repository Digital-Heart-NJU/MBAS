from __future__ import annotations
"""Standalone patient1 opening-preserving bilateral atrial-wall pipeline.
All image IO, opening selection, two-stage repair, QC, reporting, and exact
result verification are implemented in this file.  It imports no project-local
Python module.
"""
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
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
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
def clean_small_components(seg: np.ndarray, wall_min_voxels: int, cavity_min_voxels: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = seg.copy()
    rows: list[dict[str, Any]] = []
    for label_id in (WALL_LABEL, RA_LABEL, LA_LABEL):
        mask = out == label_id
        cc, n_components = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
        if n_components == 0:
            continue
        sizes = np.bincount(cc.ravel())
        if label_id == WALL_LABEL:
            rows.append(
                {
                    "label_id": label_id,
                    "label_name": LABELS[label_id],
                    "component_count_before": int(n_components),
                    "top_component_voxels_before": sorted((int(v) for v in sizes[1:] if v > 0), reverse=True)[:8],
                    "removed_voxels": 0,
                    "reason": "original biatrial wall is preserved voxelwise",
                }
            )
            continue
        largest_id = int(np.argmax(sizes[1:]) + 1)
        min_voxels = cavity_min_voxels
        keep = cc == largest_id
        for component_id in range(1, n_components + 1):
            if int(sizes[component_id]) >= min_voxels:
                keep |= cc == component_id
        removed = mask & ~keep
        out[removed] = 0
        rows.append(
            {
                "label_id": label_id,
                "label_name": LABELS[label_id],
                "component_count_before": int(n_components),
                "top_component_voxels_before": sorted((int(v) for v in sizes[1:] if v > 0), reverse=True)[:8],
                "removed_voxels": int(removed.sum()),
            }
        )
    return out, rows
def neighbor_label_counts(mask: np.ndarray, seg: np.ndarray) -> dict[str, int]:
    shell = ndi.binary_dilation(mask, structure=ndi.generate_binary_structure(3, 1)) & ~mask
    values = seg[shell]
    values = values[values > 0]
    if values.size == 0:
        return {}
    counts = np.bincount(values.astype(np.int16), minlength=max(LABELS) + 1)
    return {str(label_id): int(counts[label_id]) for label_id in (WALL_LABEL, RA_LABEL, LA_LABEL) if counts[label_id] > 0}
def fill_internal_background_holes(seg: np.ndarray, max_hole_voxels: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    background = seg == 0
    cc, n_components = ndi.label(background, structure=np.ones((3, 3, 3), dtype=bool))
    if n_components == 0:
        return seg, []
    border_ids = border_component_ids(cc)
    sizes = np.bincount(cc.ravel())
    out = seg.copy()
    rows: list[dict[str, Any]] = []
    for component_id in range(1, n_components + 1):
        if component_id in border_ids or int(sizes[component_id]) > max_hole_voxels:
            continue
        component = cc == component_id
        counts = neighbor_label_counts(component, out)
        if not counts:
            continue
        fill_label = WALL_LABEL if counts.get(str(WALL_LABEL), 0) > 0 else int(max(counts, key=counts.get))
        out[component] = fill_label
        rows.append(
            {
                "component_id": int(component_id),
                "voxels": int(sizes[component_id]),
                "fill_label": int(fill_label),
                "neighbor_label_counts": counts,
            }
        )
    return out, rows
def direct_la_background_contact(seg: np.ndarray) -> np.ndarray:
    la = seg == LA_LABEL
    background = seg == 0
    return ndi.binary_dilation(la, structure=ndi.generate_binary_structure(3, 1)) & background
def select_protected_la_openings(
    seg: np.ndarray,
    min_contact_voxels: int,
    protect_top_n: int,
    target_opening_count: int,
    selection_mode: str = "anatomic",
    pv_max_contact_voxels: int | None = None,
    pv_min_posterior_rel_y: float = 0.50,
    mv_max_rel_z: float = 0.25,
    reviewed_landmarks: list[dict[str, Any]] | None = None,
    landmark_max_assignment_distance: float = 0.15,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    contact = direct_la_background_contact(seg)
    contact_cc, n_contact = ndi.label(contact, structure=np.ones((3, 3, 3), dtype=bool))
    if n_contact == 0:
        return np.zeros(seg.shape, dtype=bool), []
    background_cc, _ = ndi.label(seg == 0, structure=np.ones((3, 3, 3), dtype=bool))
    exterior_ids = border_component_ids(background_cc)
    sizes = np.bincount(contact_cc.ravel())
    order = sorted(range(1, n_contact + 1), key=lambda component_id: int(sizes[component_id]), reverse=True)
    la_points = np.argwhere(seg == LA_LABEL)
    la_min = la_points.min(axis=0)
    la_span = np.maximum(la_points.max(axis=0) - la_min + 1, 1)
    rows: list[dict[str, Any]] = []
    eligible_rank = 0
    for rank, component_id in enumerate(order, start=1):
        component = contact_cc == component_id
        size = int(sizes[component_id])
        points = np.argwhere(component)
        background_ids = {int(value) for value in np.unique(background_cc[component]) if int(value) > 0}
        exterior_connected = bool(background_ids & exterior_ids)
        eligible = exterior_connected and size >= min_contact_voxels
        if eligible:
            eligible_rank += 1
        centroid = points.mean(axis=0)
        rows.append(
            {
                "rank": int(rank),
                "eligible_rank": int(eligible_rank) if eligible else None,
                "component_id": int(component_id),
                "contact_voxels": size,
                "contact_mm3": None,
                "exterior_connected": exterior_connected,
                "eligible_by_size_threshold": bool(eligible),
                "protected_as_anatomical_opening": False,
                "opening_class": "",
                "selection_score": None,
                "reason": "not protected",
                "centroid_zyx": [round(float(value), 3) for value in centroid],
                "relative_centroid_zyx": [round(float(value), 6) for value in ((centroid - la_min) / la_span)],
                "bbox_min_zyx": points.min(axis=0).astype(int).tolist(),
                "bbox_max_zyx": points.max(axis=0).astype(int).tolist(),
            }
        )
    protected_ids: set[int] = set()
    row_by_component = {int(row["component_id"]): row for row in rows}
    def mark_selected(component_id: int, opening_class: str, reason: str, score: float | None = None) -> None:
        protected_ids.add(component_id)
        row = row_by_component[component_id]
        row["protected_as_anatomical_opening"] = True
        row["opening_class"] = opening_class
        row["reason"] = reason
        row["selection_score"] = round(float(score), 6) if score is not None else None
    exterior_rows = [row for row in rows if bool(row["exterior_connected"])]
    if reviewed_landmarks:
        if target_opening_count > 0 and len(reviewed_landmarks) != target_opening_count:
            raise ValueError(
                f"reviewed landmark count is {len(reviewed_landmarks)}, expected {target_opening_count}"
            )
        for landmark in reviewed_landmarks:
            opening_class = str(landmark["opening_class"])
            target = np.asarray(landmark["relative_centroid_zyx"], dtype=float)
            candidates = [row for row in exterior_rows if int(row["component_id"]) not in protected_ids]
            if not candidates:
                raise ValueError(f"no unused exterior LA contact remains for reviewed landmark {opening_class}")
            def landmark_distance(row: dict[str, Any]) -> float:
                rel = np.asarray(row["relative_centroid_zyx"], dtype=float)
                return float(np.linalg.norm(rel - target))
            best = min(candidates, key=landmark_distance)
            distance = landmark_distance(best)
            if distance > landmark_max_assignment_distance:
                raise ValueError(
                    f"reviewed landmark {opening_class} has no contact within "
                    f"{landmark_max_assignment_distance:.3f} relative-coordinate units; nearest={distance:.6f}"
                )
            mark_selected(
                int(best["component_id"]),
                opening_class,
                "selected by reviewed case-local relative-coordinate landmark",
                distance,
            )
    elif selection_mode == "anatomic" and exterior_rows:
        mv_candidates = [
            row
            for row in exterior_rows
            if float(row["relative_centroid_zyx"][0]) <= mv_max_rel_z
        ]
        mv_row = max(mv_candidates or exterior_rows, key=lambda row: int(row["contact_voxels"]))
        mark_selected(
            int(mv_row["component_id"]),
            "MV",
            "selected as MV by low-z large-opening prior",
        )
        max_pv_voxels = pv_max_contact_voxels if pv_max_contact_voxels is not None else max(1, min_contact_voxels * 5)
        pv_rows = [
            row
            for row in exterior_rows
            if int(row["component_id"]) not in protected_ids
            and int(row["contact_voxels"]) <= max_pv_voxels
            and float(row["relative_centroid_zyx"][1]) >= pv_min_posterior_rel_y
        ]
        if len(pv_rows) < 4:
            pv_rows = [
                row
                for row in exterior_rows
                if int(row["component_id"]) not in protected_ids
                and int(row["contact_voxels"]) <= max_pv_voxels
                and float(row["relative_centroid_zyx"][1]) >= 0.35
            ]
        if len(pv_rows) < 4:
            pv_rows = [
                row
                for row in exterior_rows
                if int(row["component_id"]) not in protected_ids
                and int(row["contact_voxels"]) <= max_pv_voxels
            ]
        pv_targets = [
            ("PV_right_upper", np.asarray([0.78, 0.55, 0.05], dtype=float)),
            # Inferior/superior separation is primarily along the slice axis in
            # this volume. The previous right-lower target used a posterior
            # mid-slice contact and could replace the true inferior right PV.
            ("PV_right_lower", np.asarray([0.22, 0.62, 0.18], dtype=float)),
            ("PV_left_upper", np.asarray([0.70, 0.90, 0.70], dtype=float)),
            ("PV_left_lower", np.asarray([0.07, 0.67, 0.69], dtype=float)),
        ]
        scale = np.asarray([0.32, 0.28, 0.32], dtype=float)
        def spatial_opening_score(row: dict[str, Any], target: np.ndarray) -> float:
            rel = np.asarray(row["relative_centroid_zyx"], dtype=float)
            distance_score = float(np.sum(((rel - target) / scale) ** 2))
            contact_voxels = float(row["contact_voxels"])
            tiny_score = 0.1 if contact_voxels < 3 else (0.05 if contact_voxels < 5 else 0.0)
            size_score = 0.03 * math.log1p(contact_voxels)
            return distance_score + tiny_score + size_score
        for opening_class, target in pv_targets:
            candidates = [row for row in pv_rows if int(row["component_id"]) not in protected_ids]
            if not candidates:
                break
            best = min(candidates, key=lambda row: spatial_opening_score(row, target))
            mark_selected(
                int(best["component_id"]),
                opening_class,
                "selected as PV by anatomical small-opening spatial prior",
                spatial_opening_score(best, target),
            )
    elif selection_mode == "threshold":
        for row in rows:
            protect_by_size = bool(row["eligible_by_size_threshold"])
            protect_by_rank = bool(row["exterior_connected"]) and protect_top_n > 0 and int(row["rank"]) <= protect_top_n
            if protect_by_size or protect_by_rank:
                mark_selected(
                    int(row["component_id"]),
                    "opening",
                    "exterior-connected contact above size threshold" if protect_by_size else "ranked fallback",
                )
    elif target_opening_count > 0:
        for row in rows:
            eligible_rank_value = row["eligible_rank"]
            if eligible_rank_value is not None and int(eligible_rank_value) <= target_opening_count:
                mark_selected(
                    int(row["component_id"]),
                    "opening",
                    f"selected by {target_opening_count}-opening anatomical count prior",
                )
    else:
        for row in rows:
            protect_by_size = bool(row["eligible_by_size_threshold"])
            protect_by_rank = bool(row["exterior_connected"]) and protect_top_n > 0 and int(row["rank"]) <= protect_top_n
            if protect_by_size or protect_by_rank:
                mark_selected(
                    int(row["component_id"]),
                    "opening",
                    "exterior-connected contact above size threshold" if protect_by_size else "ranked fallback",
                )
    protected = np.isin(contact_cc, sorted(protected_ids)) if protected_ids else np.zeros(seg.shape, dtype=bool)
    return protected, rows
def aperture_retention_rows(
    opening_core: np.ndarray,
    original_background: np.ndarray,
    repaired_background: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    radii_mm: tuple[float, ...] = (2.0, 4.0, 6.0),
) -> list[dict[str, Any]]:
    cc, n_components = ndi.label(opening_core, structure=np.ones((3, 3, 3), dtype=bool))
    rows: list[dict[str, Any]] = []
    for component_id in range(1, n_components + 1):
        component = cc == component_id
        distance = ndi.distance_transform_edt(~component, sampling=spacing_zyx)
        row: dict[str, Any] = {
            "component_id": int(component_id),
            "core_voxels": int(component.sum()),
        }
        for radius in radii_mm:
            before = (distance <= radius) & original_background
            after = before & repaired_background
            before_voxels = int(before.sum())
            after_voxels = int(after.sum())
            row[f"retention_{radius:g}mm"] = round(float(after_voxels / max(before_voxels, 1)), 6)
            row[f"filled_{radius:g}mm"] = int(before_voxels - after_voxels)
        rows.append(row)
    return rows
def repair_left_atrial_wall(
    pred_path: Path,
    image_path: Path,
    out_path: Path,
    close_mm: float,
    opening_guard_mm: float,
    expected_shell_mm: float,
    la_search_radius_mm: float,
    left_wall_radius_mm: float,
    opening_min_contact_mm3: float,
    opening_min_contact_voxels_floor: int,
    opening_protect_top_n: int,
    target_opening_count: int,
    opening_selection_mode: str,
    pv_max_contact_mm3: float,
    pv_min_posterior_rel_y: float,
    mv_max_rel_z: float,
    output_mode: str,
    wall_min_component_voxels: int,
    cavity_min_component_voxels: int,
    max_internal_hole_voxels: int,
    reviewed_landmark_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred_vol = read_nifti(pred_path)
    image_vol = read_nifti(image_path)
    if pred_vol.shape_xyz != image_vol.shape_xyz:
        raise ValueError(f"{pred_path.name}: prediction/image shape mismatch: {pred_vol.shape_xyz} vs {image_vol.shape_xyz}")
    if tuple(round(v, 6) for v in pred_vol.spacing_xyz) != tuple(round(v, 6) for v in image_vol.spacing_xyz):
        raise ValueError(f"{pred_path.name}: prediction/image spacing mismatch: {pred_vol.spacing_xyz} vs {image_vol.spacing_xyz}")
    raw = np.rint(pred_vol.array_zyx).astype(np.uint8)
    unknown_labels = sorted(set(np.unique(raw).astype(int).tolist()) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"{pred_path.name}: unexpected labels {unknown_labels}; expected {sorted(LABELS)}")
    cleaned, cleanup_rows = clean_small_components(raw, wall_min_component_voxels, cavity_min_component_voxels)
    cleaned, internal_holes = fill_internal_background_holes(cleaned, max_internal_hole_voxels)
    background = cleaned == 0
    wall = cleaned == WALL_LABEL
    la = cleaned == LA_LABEL
    spacing_zyx = pred_vol.spacing_zyx
    opening_min_contact_voxels = min_contact_voxels_from_physical_threshold(
        spacing_zyx,
        min_contact_mm3=opening_min_contact_mm3,
        min_contact_voxels_floor=opening_min_contact_voxels_floor,
    )
    pv_max_contact_voxels = min_contact_voxels_from_physical_threshold(
        spacing_zyx,
        min_contact_mm3=pv_max_contact_mm3,
        min_contact_voxels_floor=1,
    )
    opening_core, opening_rows = select_protected_la_openings(
        cleaned,
        min_contact_voxels=opening_min_contact_voxels,
        protect_top_n=opening_protect_top_n,
        target_opening_count=target_opening_count,
        selection_mode=opening_selection_mode,
        pv_max_contact_voxels=pv_max_contact_voxels,
        pv_min_posterior_rel_y=pv_min_posterior_rel_y,
        mv_max_rel_z=mv_max_rel_z,
        reviewed_landmarks=(reviewed_landmark_spec or {}).get("openings"),
        landmark_max_assignment_distance=float((reviewed_landmark_spec or {}).get("max_assignment_distance", 0.15)),
    )
    voxel_mm3 = float(math.prod(spacing_zyx))
    for row in opening_rows:
        contact_mm3 = float(row["contact_voxels"]) * voxel_mm3
        row["contact_mm3"] = round(contact_mm3, 3)
        row["review_priority"] = opening_review_priority(
            contact_mm3,
            threshold_mm3=opening_min_contact_mm3,
            protected=bool(row["protected_as_anatomical_opening"]),
        )
    opening_guard = distance_dilation(opening_core, spacing_zyx, opening_guard_mm) & background
    near_la = distance_dilation(la, spacing_zyx, la_search_radius_mm)
    left_wall_zone = wall & distance_dilation(la, spacing_zyx, left_wall_radius_mm)
    closed_left_wall = distance_closing(left_wall_zone, spacing_zyx, close_mm)
    closed_left_complex = distance_closing(left_wall_zone | la, spacing_zyx, close_mm)
    direct_contact_holes = direct_la_background_contact(cleaned) & ~opening_guard
    expected_shell_gap = distance_dilation(la, spacing_zyx, expected_shell_mm) & background & ~opening_guard
    candidate = (((closed_left_wall | closed_left_complex) & background & near_la) | direct_contact_holes | expected_shell_gap) & ~opening_guard
    context = wall | la
    candidate_cc, n_candidate = ndi.label(candidate, structure=np.ones((3, 3, 3), dtype=bool))
    fill = np.zeros(candidate.shape, dtype=bool)
    candidate_rows: list[dict[str, Any]] = []
    sizes = np.bincount(candidate_cc.ravel()) if n_candidate else np.array([0], dtype=np.int64)
    for component_id in range(1, n_candidate + 1):
        component = candidate_cc == component_id
        touches_context = bool(np.any(ndi.binary_dilation(component, structure=np.ones((3, 3, 3), dtype=bool)) & context))
        keep = touches_context
        if keep:
            fill |= component
        candidate_rows.append(
            {
                "component_id": int(component_id),
                "voxels": int(sizes[component_id]),
                "kept": bool(keep),
                "touches_wall_or_la": touches_context,
            }
        )
    internal_wall_fill = (raw == 0) & (cleaned == WALL_LABEL) & near_la
    repair_fill = (fill | internal_wall_fill) & (raw == 0) & near_la
    repaired = raw.copy()
    repaired[repair_fill] = WALL_LABEL
    final_wall_cleanup = {
        "component_count_before": component_summary(repaired == WALL_LABEL)["component_count"],
        "top_component_voxels_before": component_summary(repaired == WALL_LABEL)["top_component_voxels"],
        "removed_voxels": 0,
        "reason": "disabled to preserve original biatrial/right-atrial wall voxels",
    }
    output_array = output_label_array(repaired, output_mode)
    write_nifti_like(pred_vol, output_array, out_path)
    repaired_background = repaired == 0
    original_background = raw == 0
    final_fill = (raw == 0) & (repaired == WALL_LABEL)
    residual_unprotected_contact = direct_la_background_contact(repaired) & ~opening_guard
    residual_expected_shell_gap = distance_dilation(repaired == LA_LABEL, spacing_zyx, expected_shell_mm) & repaired_background & ~opening_guard
    protected_fill_overlap = final_fill & opening_core
    protected_guard_overlap = final_fill & opening_guard
    aperture_rows = aperture_retention_rows(opening_core, cleaned == 0, repaired_background, spacing_zyx)
    min_retention_2mm = min((float(row["retention_2mm"]) for row in aperture_rows), default=1.0)
    min_retention_4mm = min((float(row["retention_4mm"]) for row in aperture_rows), default=1.0)
    cavity_label_loss = {
        str(label_id): int(np.count_nonzero((raw == label_id) & (repaired != label_id)))
        for label_id in (RA_LABEL, LA_LABEL)
    }
    background_to_cavity = {
        str(label_id): int(np.count_nonzero((raw == 0) & (repaired == label_id)))
        for label_id in (RA_LABEL, LA_LABEL)
    }
    wall_to_cavity = {
        str(label_id): int(np.count_nonzero((raw == WALL_LABEL) & (repaired == label_id)))
        for label_id in (RA_LABEL, LA_LABEL)
    }
    original_wall_lost = int(np.count_nonzero((raw == WALL_LABEL) & (repaired != WALL_LABEL)))
    output_original_wall_lost = int(np.count_nonzero((raw == WALL_LABEL) & (output_array != WALL_LABEL)))
    protected_opening_count = int(sum(1 for row in opening_rows if row["protected_as_anatomical_opening"]))
    failures: list[str] = []
    warnings: list[str] = []
    if target_opening_count > 0 and protected_opening_count != target_opening_count:
        failures.append(f"protected opening count is {protected_opening_count}, expected {target_opening_count}")
    if original_wall_lost > 0:
        failures.append(f"original biatrial wall voxels were overwritten: {original_wall_lost}")
    if output_original_wall_lost > 0:
        failures.append(f"original biatrial wall voxels missing from output: {output_original_wall_lost}")
    if int(residual_unprotected_contact.sum()) > 0:
        failures.append(f"residual unprotected LA-background contact voxels: {int(residual_unprotected_contact.sum())}")
    if int(residual_expected_shell_gap.sum()) > 0:
        failures.append(f"residual expected-shell gap voxels: {int(residual_expected_shell_gap.sum())}")
    if int(protected_fill_overlap.sum()) > 0:
        failures.append(f"fill overlaps protected opening core: {int(protected_fill_overlap.sum())}")
    if int(protected_guard_overlap.sum()) > 0:
        failures.append(f"fill overlaps protected opening guard: {int(protected_guard_overlap.sum())}")
    if any(count > 0 for count in cavity_label_loss.values()):
        failures.append(f"original RA/LA cavity voxels were overwritten: {cavity_label_loss}")
    if int(opening_core.sum()) == 0:
        failures.append("no protected LA anatomical opening candidates detected")
    if min_retention_2mm < 0.98:
        failures.append(f"minimum 2mm opening aperture retention is low: {min_retention_2mm:.3f}")
    if min_retention_4mm < 0.90:
        warnings.append(f"minimum 4mm opening aperture retention is low: {min_retention_4mm:.3f}")
    status = "fail" if failures else ("warning" if warnings else "pass")
    changed = raw != repaired
    return {
        "case": strip_nifti_suffix(pred_path),
        "status": status,
        "input_prediction": str(pred_path),
        "input_image": str(image_path),
        "output": str(out_path),
        "parameters": {
            "close_mm": close_mm,
            "opening_guard_mm": opening_guard_mm,
            "expected_shell_mm": expected_shell_mm,
            "la_search_radius_mm": la_search_radius_mm,
            "left_wall_radius_mm": left_wall_radius_mm,
            "opening_min_contact_mm3": opening_min_contact_mm3,
            "opening_min_contact_voxels": opening_min_contact_voxels,
            "opening_min_contact_voxels_floor": opening_min_contact_voxels_floor,
            "opening_protect_top_n": opening_protect_top_n,
            "target_opening_count": target_opening_count,
            "opening_selection_mode": opening_selection_mode,
            "pv_max_contact_mm3": pv_max_contact_mm3,
            "pv_max_contact_voxels": pv_max_contact_voxels,
            "pv_min_posterior_rel_y": pv_min_posterior_rel_y,
            "mv_max_rel_z": mv_max_rel_z,
            "reviewed_landmarks_applied": bool(reviewed_landmark_spec),
            "reviewed_landmark_source": (reviewed_landmark_spec or {}).get("source"),
            "output_mode": output_mode,
            "wall_min_component_voxels": wall_min_component_voxels,
            "cavity_min_component_voxels": cavity_min_component_voxels,
            "max_internal_hole_voxels": max_internal_hole_voxels,
        },
        "geometry": {
            "shape_xyz": pred_vol.shape_xyz,
            "spacing_xyz": pred_vol.spacing_xyz,
        },
        "label_counts_before": label_counts(raw),
        "label_counts_after_semantic": label_counts(repaired),
        "label_counts_output": label_counts(output_array),
        "cleanup": cleanup_rows,
        "internal_background_holes_filled": internal_holes,
        "final_wall_cleanup": final_wall_cleanup,
        "opening_components": opening_rows,
        "candidate_components": sorted(candidate_rows, key=lambda row: int(row["voxels"]), reverse=True)[:20],
        "aperture_retention": aperture_rows,
        "qc": {
            "changed_voxels": int(changed.sum()),
            "background_to_wall_voxels": int(np.count_nonzero((raw == 0) & (repaired == WALL_LABEL))),
            "wall_to_background_cleanup_voxels": int(np.count_nonzero((raw == WALL_LABEL) & (repaired == 0))),
            "wall_to_ra_cleanup_voxels": wall_to_cavity[str(RA_LABEL)],
            "wall_to_la_cleanup_voxels": wall_to_cavity[str(LA_LABEL)],
            "background_to_ra_voxels": background_to_cavity[str(RA_LABEL)],
            "background_to_la_voxels": background_to_cavity[str(LA_LABEL)],
            "candidate_fill_voxels_before_final_cleanup": int(fill.sum()),
            "internal_wall_hole_fill_voxels": int(internal_wall_fill.sum()),
            "final_wall_cleanup_removed_voxels": int(final_wall_cleanup["removed_voxels"]),
            "original_wall_lost_voxels": original_wall_lost,
            "output_original_wall_lost_voxels": output_original_wall_lost,
            "original_ra_cavity_lost_voxels": cavity_label_loss[str(RA_LABEL)],
            "original_la_cavity_lost_voxels": cavity_label_loss[str(LA_LABEL)],
            "protected_opening_core_voxels": int(opening_core.sum()),
            "protected_opening_components": protected_opening_count,
            "opening_guard_voxels": int(opening_guard.sum()),
            "fill_voxels": int(final_fill.sum()),
            "fill_overlap_protected_core_voxels": int(protected_fill_overlap.sum()),
            "fill_overlap_protected_guard_voxels": int(protected_guard_overlap.sum()),
            "residual_unprotected_la_contact_voxels": int(residual_unprotected_contact.sum()),
            "residual_unprotected_la_contact_components": component_summary(residual_unprotected_contact)["component_count"],
            "residual_expected_shell_gap_voxels": int(residual_expected_shell_gap.sum()),
            "residual_expected_shell_gap_components": component_summary(residual_expected_shell_gap)["component_count"],
            "wall_components_after": component_summary(repaired == WALL_LABEL),
            "ra_components_after": component_summary(repaired == RA_LABEL),
            "la_components_after": component_summary(repaired == LA_LABEL),
            "min_opening_retention_2mm": round(float(min_retention_2mm), 6),
            "min_opening_retention_4mm": round(float(min_retention_4mm), 6),
        },
        "failures": failures,
        "warnings": warnings,
    }
def write_stage1_reports(report_dir: Path, reports: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_json = report_dir / "postprocess_private_pred073_summary.json"
    summary_json.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_csv = report_dir / "postprocess_private_pred073_summary.csv"
    fields = [
        "case",
        "status",
        "output",
        "output_mode",
        "opening_selection_mode",
        "reviewed_landmarks_applied",
        "target_opening_count",
        "fill_voxels",
        "background_to_wall_voxels",
        "background_to_ra_voxels",
        "background_to_la_voxels",
        "wall_to_background_cleanup_voxels",
        "wall_to_ra_cleanup_voxels",
        "wall_to_la_cleanup_voxels",
        "final_wall_cleanup_removed_voxels",
        "original_wall_lost_voxels",
        "output_original_wall_lost_voxels",
        "original_ra_cavity_lost_voxels",
        "original_la_cavity_lost_voxels",
        "protected_opening_components",
        "protected_opening_core_voxels",
        "opening_guard_voxels",
        "residual_unprotected_la_contact_voxels",
        "residual_expected_shell_gap_voxels",
        "wall_components_after",
        "min_opening_retention_2mm",
        "min_opening_retention_4mm",
        "failure_count",
        "warning_count",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            qc = report["qc"]
            writer.writerow(
                {
                    "case": report["case"],
                    "status": report["status"],
                    "output": report["output"],
                    "output_mode": report["parameters"]["output_mode"],
                    "opening_selection_mode": report["parameters"]["opening_selection_mode"],
                    "reviewed_landmarks_applied": report["parameters"]["reviewed_landmarks_applied"],
                    "target_opening_count": report["parameters"]["target_opening_count"],
                    "fill_voxels": qc["fill_voxels"],
                    "background_to_wall_voxels": qc["background_to_wall_voxels"],
                    "background_to_ra_voxels": qc["background_to_ra_voxels"],
                    "background_to_la_voxels": qc["background_to_la_voxels"],
                    "wall_to_background_cleanup_voxels": qc["wall_to_background_cleanup_voxels"],
                    "wall_to_ra_cleanup_voxels": qc["wall_to_ra_cleanup_voxels"],
                    "wall_to_la_cleanup_voxels": qc["wall_to_la_cleanup_voxels"],
                    "final_wall_cleanup_removed_voxels": qc["final_wall_cleanup_removed_voxels"],
                    "original_wall_lost_voxels": qc["original_wall_lost_voxels"],
                    "output_original_wall_lost_voxels": qc["output_original_wall_lost_voxels"],
                    "original_ra_cavity_lost_voxels": qc["original_ra_cavity_lost_voxels"],
                    "original_la_cavity_lost_voxels": qc["original_la_cavity_lost_voxels"],
                    "protected_opening_components": qc["protected_opening_components"],
                    "protected_opening_core_voxels": qc["protected_opening_core_voxels"],
                    "opening_guard_voxels": qc["opening_guard_voxels"],
                    "residual_unprotected_la_contact_voxels": qc["residual_unprotected_la_contact_voxels"],
                    "residual_expected_shell_gap_voxels": qc["residual_expected_shell_gap_voxels"],
                    "wall_components_after": qc["wall_components_after"]["component_count"],
                    "min_opening_retention_2mm": qc["min_opening_retention_2mm"],
                    "min_opening_retention_4mm": qc["min_opening_retention_4mm"],
                    "failure_count": len(report["failures"]),
                    "warning_count": len(report["warnings"]),
                }
            )
    opening_csv = report_dir / "postprocess_private_pred073_opening_components.csv"
    opening_fields = [
        "case",
        "rank",
        "eligible_rank",
        "component_id",
        "contact_voxels",
        "contact_mm3",
        "exterior_connected",
        "eligible_by_size_threshold",
        "protected_as_anatomical_opening",
        "opening_class",
        "selection_score",
        "reason",
        "review_priority",
        "relative_centroid_zyx",
        "centroid_zyx",
        "bbox_min_zyx",
        "bbox_max_zyx",
    ]
    with opening_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=opening_fields)
        writer.writeheader()
        for report in reports:
            for row in report["opening_components"]:
                writer.writerow(
                    {
                        "case": report["case"],
                        "rank": row["rank"],
                        "eligible_rank": row["eligible_rank"],
                        "component_id": row["component_id"],
                        "contact_voxels": row["contact_voxels"],
                        "contact_mm3": row["contact_mm3"],
                        "exterior_connected": row["exterior_connected"],
                        "eligible_by_size_threshold": row["eligible_by_size_threshold"],
                        "protected_as_anatomical_opening": row["protected_as_anatomical_opening"],
                        "opening_class": row["opening_class"],
                        "selection_score": row["selection_score"],
                        "reason": row["reason"],
                        "review_priority": row["review_priority"],
                        "relative_centroid_zyx": json.dumps(row["relative_centroid_zyx"]),
                        "centroid_zyx": json.dumps(row["centroid_zyx"]),
                        "bbox_min_zyx": json.dumps(row["bbox_min_zyx"]),
                        "bbox_max_zyx": json.dumps(row["bbox_max_zyx"]),
                    }
                )
"""Additive bilateral atrial-wall microhole repair.
This is an independent post-processing stage built on top of the existing
opening-correct wall result.  It imports the stable baseline utilities but does
not modify the original post-processing script or its outputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
REPRO = Path(__file__).resolve().parents[1]
DEFAULT_PRED_DIR = REPRO / "private_pred073"
DEFAULT_IMAGE_DIR = REPRO / "private_input073"
DEFAULT_BASELINE_DIR = REPRO / "private_pred073_repaired_recovered_20260922"
DEFAULT_BASELINE_REPORT = (
    REPRO
    / "private_pred073_repaired_reports_recovered_20260922"
    / "postprocess_private_pred073_summary.json"
)
DEFAULT_OUTPUT_DIR = REPRO / "private_pred073_repaired_biatrial_holes"
DEFAULT_REPORT_DIR = REPRO / "private_pred073_repaired_biatrial_holes_reports"
DEFAULT_KNOWN_WRONG_DIRS = (
    REPRO / "private_pred073_repaired_final",
    REPRO / "private_pred073_repaired_final_best_20260908_mv_pv4",
)
FULL_STRUCTURE = np.ones((3, 3, 3), dtype=bool)
FACE_STRUCTURE = ndi.generate_binary_structure(3, 1)
def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array.astype(np.uint8, copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
def find_case_nifti(directory: Path, case: str) -> Path:
    for path in (directory / f"{case}.nii.gz", directory / f"{case}.nii"):
        if path.exists():
            return path
    raise FileNotFoundError(f"No NIfTI for {case} in {directory}")
def validate_geometry(
    reference: NiftiVolume,
    other: NiftiVolume,
    reference_name: str,
    other_name: str,
) -> None:
    if reference.shape_xyz != other.shape_xyz:
        raise ValueError(
            f"geometry mismatch: {reference_name} shape {reference.shape_xyz} != "
            f"{other_name} shape {other.shape_xyz}"
        )
    if not np.allclose(reference.spacing_xyz, other.spacing_xyz, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"geometry mismatch: {reference_name} spacing {reference.spacing_xyz} != "
            f"{other_name} spacing {other.spacing_xyz}"
        )
    if not np.allclose(reference.origin_xyz, other.origin_xyz, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"geometry mismatch: {reference_name} origin {reference.origin_xyz} != "
            f"{other_name} origin {other.origin_xyz}"
        )
def load_reports(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")
    reports: dict[str, dict[str, Any]] = {}
    for report in payload:
        case = str(report.get("case", ""))
        if not case or case in reports:
            raise ValueError(f"{path}: missing or duplicate case {case!r}")
        reports[case] = report
    return reports
def compose_semantic(raw: np.ndarray, wall: np.ndarray) -> np.ndarray:
    semantic = raw.copy()
    semantic[(raw == 0) & wall] = WALL_LABEL
    return semantic
def exterior_background(semantic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    background = semantic == 0
    background_cc, _ = ndi.label(background, structure=FULL_STRUCTURE)
    exterior_ids = border_component_ids(background_cc)
    exterior = np.isin(background_cc, sorted(exterior_ids))
    return background, exterior
def chamber_exterior_contact(
    semantic: np.ndarray,
    chamber_label: int,
    background: np.ndarray,
    exterior: np.ndarray,
) -> np.ndarray:
    return ndi.binary_dilation(semantic == chamber_label, structure=FACE_STRUCTURE) & background & exterior
def reviewed_la_opening_core(raw: np.ndarray, baseline_report: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cleaned, _ = clean_small_components(raw, wall_min_voxels=500, cavity_min_voxels=500)
    cleaned, _ = fill_internal_background_holes(cleaned, max_hole_voxels=20000)
    contact = direct_la_background_contact(cleaned)
    contact_cc, _ = ndi.label(contact, structure=FULL_STRUCTURE)
    selected_rows = [
        row
        for row in baseline_report["opening_components"]
        if bool(row["protected_as_anatomical_opening"])
    ]
    if len(selected_rows) != 5:
        raise ValueError(f"expected five reviewed LA openings, found {len(selected_rows)}")
    selected_ids = [int(row["component_id"]) for row in selected_rows]
    if any(component_id <= 0 or not np.any(contact_cc == component_id) for component_id in selected_ids):
        raise ValueError("reviewed LA opening IDs do not match the raw prediction")
    return np.isin(contact_cc, selected_ids), selected_rows
def select_ra_opening_core(
    semantic: np.ndarray,
    target_count: int,
    min_contact_voxels: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    background, exterior = exterior_background(semantic)
    contact = chamber_exterior_contact(semantic, RA_LABEL, background, exterior)
    contact_cc, count = ndi.label(contact, structure=FULL_STRUCTURE)
    sizes = np.bincount(contact_cc.ravel()) if count else np.asarray([0], dtype=np.int64)
    order = sorted(range(1, count + 1), key=lambda component_id: int(sizes[component_id]), reverse=True)
    selected_ids = [
        component_id
        for component_id in order
        if int(sizes[component_id]) >= min_contact_voxels
    ][:target_count]
    if len(selected_ids) != target_count:
        raise ValueError(
            f"found {len(selected_ids)} RA opening contacts >= {min_contact_voxels} voxels; "
            f"expected {target_count}"
        )
    rows: list[dict[str, Any]] = []
    for rank, component_id in enumerate(order, start=1):
        component = contact_cc == component_id
        points = np.argwhere(component)
        rows.append(
            {
                "rank": rank,
                "component_id": int(component_id),
                "contact_voxels": int(sizes[component_id]),
                "protected_as_anatomical_opening": component_id in selected_ids,
                "centroid_zyx": [round(float(value), 3) for value in points.mean(axis=0)],
                "bbox_min_zyx": points.min(axis=0).astype(int).tolist(),
                "bbox_max_zyx": points.max(axis=0).astype(int).tolist(),
            }
        )
    return np.isin(contact_cc, selected_ids), rows
def load_known_wrong_protection(
    case: str,
    baseline_wall: np.ndarray,
    baseline_vol: NiftiVolume,
    variant_dirs: list[Path],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    protection = np.zeros(baseline_wall.shape, dtype=bool)
    rows: list[dict[str, Any]] = []
    for directory in variant_dirs:
        variant_path = find_case_nifti(directory, case)
        variant_vol = read_nifti(variant_path)
        validate_geometry(baseline_vol, variant_vol, "baseline", f"known wrong variant {variant_path}")
        variant = np.rint(variant_vol.array_zyx).astype(np.uint8)
        unexpected = sorted(set(np.unique(variant).astype(int).tolist()) - {0, WALL_LABEL})
        if unexpected:
            raise ValueError(f"{variant_path}: expected wall-only labels 0/1, found {unexpected}")
        wrong_fill = (variant == WALL_LABEL) & ~baseline_wall
        protection |= wrong_fill
        rows.append(
            {
                "path": str(variant_path),
                "wrong_fill_voxels_relative_to_baseline": int(wrong_fill.sum()),
            }
        )
    return protection, rows
def half_neighborhood_directions() -> list[tuple[int, int, int]]:
    directions: list[tuple[int, int, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                direction = (dz, dy, dx)
                if direction == (0, 0, 0):
                    continue
                for value in direction:
                    if value:
                        if value > 0:
                            directions.append(direction)
                        break
    if len(directions) != 13:
        raise AssertionError(f"expected 13 opposing direction pairs, found {len(directions)}")
    return directions
def sample_offset(mask: np.ndarray, direction: tuple[int, int, int]) -> np.ndarray:
    """Sample mask at index + direction, padding outside the volume with false."""
    out = np.zeros_like(mask, dtype=bool)
    source: list[slice] = []
    target: list[slice] = []
    for delta in direction:
        if delta > 0:
            source.append(slice(delta, None))
            target.append(slice(None, -delta))
        elif delta < 0:
            source.append(slice(None, delta))
            target.append(slice(-delta, None))
        else:
            source.append(slice(None))
            target.append(slice(None))
    out[tuple(target)] = mask[tuple(source)]
    return out
def directionally_bracketed_background(wall: np.ndarray) -> np.ndarray:
    bracketed = np.zeros(wall.shape, dtype=bool)
    for direction in half_neighborhood_directions():
        opposite = tuple(-value for value in direction)
        bracketed |= sample_offset(wall, direction) & sample_offset(wall, opposite)
    return bracketed & ~wall
def keep_small_components(mask: np.ndarray, max_voxels: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels, count = ndi.label(mask, structure=FULL_STRUCTURE)
    sizes = np.bincount(labels.ravel()) if count else np.asarray([0], dtype=np.int64)
    keep_ids = [component_id for component_id in range(1, count + 1) if int(sizes[component_id]) <= max_voxels]
    kept = np.isin(labels, keep_ids) if keep_ids else np.zeros(mask.shape, dtype=bool)
    rows: list[dict[str, Any]] = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        points = np.argwhere(component)
        rows.append(
            {
                "component_id": int(component_id),
                "voxels": int(sizes[component_id]),
                "accepted": component_id in keep_ids,
                "centroid_zyx": [round(float(value), 3) for value in points.mean(axis=0)],
                "bbox_min_zyx": points.min(axis=0).astype(int).tolist(),
                "bbox_max_zyx": points.max(axis=0).astype(int).tolist(),
            }
        )
    rows.sort(key=lambda row: (-int(row["voxels"]), int(row["component_id"])))
    return kept, rows
def matrix_array(matrix: Any) -> np.ndarray | None:
    if matrix is None:
        return None
    return np.asarray([[matrix.GetElement(row, column) for column in range(4)] for row in range(4)])
def same_matrix(first: Any, second: Any) -> bool:
    first_array = matrix_array(first)
    second_array = matrix_array(second)
    if first_array is None or second_array is None:
        return first_array is None and second_array is None
    return bool(np.allclose(first_array, second_array, rtol=0.0, atol=1e-7))
def plane(array: np.ndarray, axis: int, index: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if axis == 0:
        return array[index, lo[1] : hi[1], lo[2] : hi[2]]
    if axis == 1:
        return array[lo[0] : hi[0], index, lo[2] : hi[2]]
    return array[lo[0] : hi[0], lo[1] : hi[1], index]
def write_qc_montage(
    image: np.ndarray,
    baseline_wall: np.ndarray,
    final_wall: np.ndarray,
    additions_by_stage: np.ndarray,
    opening_core: np.ndarray,
    known_wrong: np.ndarray,
    out_path: Path,
) -> None:
    focus = baseline_wall | (additions_by_stage > 0) | opening_core
    points = np.argwhere(focus)
    lo = np.maximum(points.min(axis=0) - 18, 0)
    hi = np.minimum(points.max(axis=0) + 19, focus.shape)
    finite = image[np.isfinite(image)]
    positive = finite[finite > 0]
    window = positive if positive.size else finite
    vmin, vmax = np.percentile(window, [1, 99.5])
    if vmax <= vmin:
        vmax = vmin + 1.0
    rows: list[tuple[str, int, int]] = []
    additions = additions_by_stage > 0
    for name, axis in (("axial", 0), ("coronal", 1), ("sagittal", 2)):
        sum_axes = tuple(value for value in range(3) if value != axis)
        counts = additions.sum(axis=sum_axes)
        indices = np.argsort(counts)[-2:]
        for index in sorted({int(value) for value in indices if counts[int(value)] > 0}):
            rows.append((name, axis, index))
    fig, axes = plt.subplots(len(rows), 3, figsize=(11.5, max(4.0, len(rows) * 2.35)))
    axes = np.atleast_2d(axes)
    for row_index, (name, axis, index) in enumerate(rows):
        base_image = plane(image, axis, index, lo, hi)
        baseline_2d = plane(baseline_wall, axis, index, lo, hi)
        final_2d = plane(final_wall, axis, index, lo, hi)
        stage_2d = plane(additions_by_stage, axis, index, lo, hi)
        opening_2d = plane(opening_core, axis, index, lo, hi)
        wrong_2d = plane(known_wrong, axis, index, lo, hi)
        for column, title in enumerate(("baseline", "additions by pass", "final wall")):
            ax = axes[row_index, column]
            ax.imshow(base_image, cmap="gray", vmin=vmin, vmax=vmax)
            rgba = np.zeros((*base_image.shape, 4), dtype=float)
            wall_2d = baseline_2d if column != 2 else final_2d
            rgba[wall_2d, :3] = (1.0, 0.1, 0.05)
            rgba[wall_2d, 3] = 0.38
            if column == 1:
                rgba[stage_2d == 1, :3] = (0.0, 1.0, 0.2)
                rgba[stage_2d == 1, 3] = 0.92
                rgba[stage_2d == 2, :3] = (1.0, 0.85, 0.0)
                rgba[stage_2d == 2, 3] = 0.98
                rgba[stage_2d == 3, :3] = (0.15, 0.55, 1.0)
                rgba[stage_2d == 3, 3] = 0.98
                rgba[wrong_2d, :3] = (1.0, 0.0, 1.0)
                rgba[wrong_2d, 3] = 0.72
                rgba[opening_2d, :3] = (0.0, 1.0, 1.0)
                rgba[opening_2d, 3] = 0.78
            ax.imshow(rgba)
            ax.set_title(f"{name} {index}: {title}", fontsize=9)
            ax.axis("off")
    fig.suptitle(
        "red=wall; green=3mm shell; yellow=gap pass 1; blue=gap pass 2; cyan=openings; magenta=wrong-fill exclusion",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
def repair_case(
    case: str,
    pred_dir: Path,
    image_dir: Path,
    baseline_dir: Path,
    baseline_report_path: Path,
    baseline_report: dict[str, Any],
    known_wrong_dirs: list[Path],
    output_dir: Path,
    report_dir: Path,
    passes: int,
    shell_mm: float,
    opening_guard_mm: float,
    wall_search_mm: float,
    max_hole_component_voxels: int,
    ra_opening_count: int,
    ra_opening_min_contact_voxels: int,
    max_total_additions: int,
    overwrite: bool,
) -> dict[str, Any]:
    pred_path = find_case_nifti(pred_dir, case)
    baseline_path = find_case_nifti(baseline_dir, case)
    image_path = matched_image_path(pred_path, image_dir)
    out_path = output_dir / f"{case}.nii.gz"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; rerun with --overwrite")
    pred_vol = read_nifti(pred_path)
    baseline_vol = read_nifti(baseline_path)
    image_vol = read_nifti(image_path)
    validate_geometry(pred_vol, baseline_vol, "prediction", "baseline")
    validate_geometry(pred_vol, image_vol, "prediction", "image")
    raw = np.rint(pred_vol.array_zyx).astype(np.uint8)
    baseline_array = np.rint(baseline_vol.array_zyx).astype(np.uint8)
    if sorted(set(np.unique(raw).astype(int).tolist()) - set(LABELS)):
        raise ValueError(f"{pred_path}: unexpected prediction labels")
    if sorted(set(np.unique(baseline_array).astype(int).tolist()) - {0, WALL_LABEL}):
        raise ValueError(f"{baseline_path}: expected a wall-only 0/1 label map")
    baseline_wall = baseline_array == WALL_LABEL
    if np.any((raw == WALL_LABEL) & ~baseline_wall):
        raise ValueError("baseline is missing original wall voxels")
    if np.any(baseline_wall & ((raw == RA_LABEL) | (raw == LA_LABEL))):
        raise ValueError("baseline wall overlaps original RA/LA blood pool")
    baseline_semantic = compose_semantic(raw, baseline_wall)
    la_core, la_rows = reviewed_la_opening_core(raw, baseline_report)
    ra_core, ra_rows = select_ra_opening_core(
        baseline_semantic,
        target_count=ra_opening_count,
        min_contact_voxels=ra_opening_min_contact_voxels,
    )
    opening_core = la_core | ra_core
    baseline_background = baseline_semantic == 0
    opening_guard = distance_dilation(opening_core, pred_vol.spacing_zyx, opening_guard_mm) & baseline_background
    known_wrong, known_wrong_rows = load_known_wrong_protection(
        case,
        baseline_wall,
        baseline_vol,
        known_wrong_dirs,
    )
    atrial_cavities = (raw == RA_LABEL) | (raw == LA_LABEL)
    near_biatrial = distance_dilation(atrial_cavities, pred_vol.spacing_zyx, wall_search_mm)
    shell_candidate = (
        distance_dilation(atrial_cavities, pred_vol.spacing_zyx, shell_mm)
        & (raw == 0)
        & ~baseline_wall
        & ~opening_guard
        & ~known_wrong
    )
    current_wall = baseline_wall | shell_candidate
    additions_by_stage = np.zeros(raw.shape, dtype=np.uint8)
    additions_by_stage[shell_candidate] = 1
    stage_rows: list[dict[str, Any]] = [
        {
            "stage": "biatrial_shell",
            "stage_id": 1,
            "candidate_voxels": int(shell_candidate.sum()),
            "accepted_voxels": int(shell_candidate.sum()),
            "RA_exterior_leak_accepted_voxels": 0,
            "LA_exterior_leak_accepted_voxels": 0,
            "new_union_voxels": int(shell_candidate.sum()),
            "cumulative_added_voxels": int(shell_candidate.sum()),
        }
    ]
    component_rows: list[dict[str, Any]] = []
    for pass_number in range(1, passes + 1):
        semantic = compose_semantic(raw, current_wall)
        background, exterior = exterior_background(semantic)
        eligible = (raw == 0) & ~current_wall & near_biatrial & ~opening_guard & ~known_wrong
        bracketed = directionally_bracketed_background(current_wall) & eligible
        microgap_fill, microgap_rows = keep_small_components(bracketed, max_hole_component_voxels)
        for row in microgap_rows:
            row.update({"pass": pass_number, "source": "directionally_bracketed"})
            component_rows.append(row)
        leak_fill = np.zeros(raw.shape, dtype=bool)
        leak_counts: dict[str, int] = {}
        for chamber_label, chamber_name in ((RA_LABEL, "RA"), (LA_LABEL, "LA")):
            contact = chamber_exterior_contact(semantic, chamber_label, background, exterior) & eligible
            chamber_fill, chamber_rows = keep_small_components(contact, max_hole_component_voxels)
            leak_fill |= chamber_fill
            leak_counts[chamber_name] = int(chamber_fill.sum())
            for row in chamber_rows:
                row.update({"pass": pass_number, "source": f"{chamber_name}_exterior_leak"})
                component_rows.append(row)
        additions = (microgap_fill | leak_fill) & ~current_wall
        stage_id = pass_number + 1
        additions_by_stage[additions] = stage_id
        current_wall |= additions
        stage_rows.append(
            {
                "stage": f"directional_pass_{pass_number}",
                "stage_id": stage_id,
                "candidate_voxels": int(bracketed.sum()),
                "accepted_voxels": int(microgap_fill.sum()),
                "RA_exterior_leak_accepted_voxels": leak_counts["RA"],
                "LA_exterior_leak_accepted_voxels": leak_counts["LA"],
                "new_union_voxels": int(additions.sum()),
                "cumulative_added_voxels": int((additions_by_stage > 0).sum()),
            }
        )
        if not np.any(additions):
            break
    final_wall = current_wall
    output_array = final_wall.astype(np.uint8)
    added = final_wall & ~baseline_wall
    lost = baseline_wall & ~final_wall
    final_semantic = compose_semantic(raw, final_wall)
    final_background, final_exterior = exterior_background(final_semantic)
    opening_cc, opening_count = ndi.label(opening_core, structure=FULL_STRUCTURE)
    exterior_background_cc, _ = ndi.label(final_background, structure=FULL_STRUCTURE)
    final_exterior_ids = border_component_ids(exterior_background_cc)
    opening_exterior_rows: list[dict[str, Any]] = []
    for component_id in range(1, opening_count + 1):
        component = opening_cc == component_id
        background_ids = {
            int(value)
            for value in np.unique(exterior_background_cc[component])
            if int(value) > 0
        }
        opening_exterior_rows.append(
            {
                "component_id": component_id,
                "core_voxels": int(component.sum()),
                "remains_background": bool(np.all(final_background[component])),
                "remains_exterior_connected": bool(background_ids & final_exterior_ids),
            }
        )
    aperture_rows = aperture_retention_rows(
        opening_core,
        baseline_background,
        final_background,
        pred_vol.spacing_zyx,
    )
    min_retention_2mm = min((float(row["retention_2mm"]) for row in aperture_rows), default=1.0)
    min_retention_4mm = min((float(row["retention_4mm"]) for row in aperture_rows), default=1.0)
    failures: list[str] = []
    if int(lost.sum()) != 0:
        failures.append(f"baseline wall voxels lost: {int(lost.sum())}")
    if int((added & (raw != 0)).sum()) != 0:
        failures.append("added wall overlaps raw RA/LA/wall labels")
    if int((added & opening_core).sum()) != 0 or int((added & opening_guard).sum()) != 0:
        failures.append("added wall overlaps protected opening core or guard")
    if int((added & known_wrong).sum()) != 0:
        failures.append("added wall overlaps a known historical wrong-fill region")
    if int(added.sum()) > max_total_additions:
        failures.append(f"total additions {int(added.sum())} exceed limit {max_total_additions}")
    if any(not row["remains_background"] or not row["remains_exterior_connected"] for row in opening_exterior_rows):
        failures.append("one or more protected openings lost background/exterior connectivity")
    if min_retention_2mm < 1.0:
        failures.append(f"minimum 2 mm opening retention is {min_retention_2mm:.6f}, expected 1.0")
    if min_retention_4mm < 1.0:
        failures.append(f"minimum 4 mm opening retention is {min_retention_4mm:.6f}, expected 1.0")
    if failures:
        raise RuntimeError("; ".join(failures))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    additions_path = report_dir / f"{case}_additions_by_stage.nii.gz"
    opening_path = report_dir / f"{case}_protected_openings.nii.gz"
    known_wrong_path = report_dir / f"{case}_known_wrong_fill_protection.nii.gz"
    montage_path = report_dir / f"{case}_biatrial_holes_qc.png"
    write_nifti_like(baseline_vol, output_array, out_path)
    write_nifti_like(baseline_vol, additions_by_stage, additions_path)
    write_nifti_like(baseline_vol, opening_core.astype(np.uint8), opening_path)
    write_nifti_like(baseline_vol, known_wrong.astype(np.uint8), known_wrong_path)
    write_qc_montage(
        image=image_vol.array_zyx.astype(float),
        baseline_wall=baseline_wall,
        final_wall=final_wall,
        additions_by_stage=additions_by_stage,
        opening_core=opening_core,
        known_wrong=known_wrong,
        out_path=montage_path,
    )
    ra_distance = ndi.distance_transform_edt(raw != RA_LABEL, sampling=pred_vol.spacing_zyx)
    la_distance = ndi.distance_transform_edt(raw != LA_LABEL, sampling=pred_vol.spacing_zyx)
    report = {
        "case": case,
        "status": "pass",
        "method": "additive bilateral microhole repair after the opening-correct baseline",
        "inputs": {
            "prediction": str(pred_path),
            "image": str(image_path),
            "baseline_wall": str(baseline_path),
            "baseline_report": str(baseline_report_path),
            "known_wrong_variants": known_wrong_rows,
        },
        "outputs": {
            "repaired_wall": str(out_path),
            "additions_by_stage": str(additions_path),
            "protected_openings": str(opening_path),
            "known_wrong_fill_protection": str(known_wrong_path),
            "qc_montage": str(montage_path),
        },
        "parameters": {
            "passes": passes,
            "shell_mm": shell_mm,
            "opening_guard_mm": opening_guard_mm,
            "wall_search_mm": wall_search_mm,
            "max_hole_component_voxels": max_hole_component_voxels,
            "ra_opening_count": ra_opening_count,
            "ra_opening_min_contact_voxels": ra_opening_min_contact_voxels,
            "max_total_additions": max_total_additions,
            "direction_pairs": 13,
            "policy": "add only; bilateral RA/LA; reviewed openings, blood pools, and known wrong fills immutable",
        },
        "reviewed_LA_openings": [
            {
                "opening_class": row["opening_class"],
                "component_id": int(row["component_id"]),
                "contact_voxels": int(row["contact_voxels"]),
            }
            for row in la_rows
        ],
        "RA_opening_contacts": ra_rows,
        "stages": stage_rows,
        "component_rows": component_rows,
        "aperture_retention": aperture_rows,
        "opening_exterior_connectivity": opening_exterior_rows,
        "qc": {
            "baseline_wall_voxels": int(baseline_wall.sum()),
            "final_wall_voxels": int(final_wall.sum()),
            "added_voxels": int(added.sum()),
            "added_nearer_RA_voxels": int((added & (ra_distance <= la_distance)).sum()),
            "added_nearer_LA_voxels": int((added & (la_distance < ra_distance)).sum()),
            "baseline_wall_lost_voxels": int(lost.sum()),
            "added_overlap_raw_nonbackground_voxels": int((added & (raw != 0)).sum()),
            "added_overlap_opening_core_voxels": int((added & opening_core).sum()),
            "added_overlap_opening_guard_voxels": int((added & opening_guard).sum()),
            "added_overlap_known_wrong_fill_voxels": int((added & known_wrong).sum()),
            "protected_LA_opening_count": 5,
            "protected_RA_opening_count": int(sum(row["protected_as_anatomical_opening"] for row in ra_rows)),
            "protected_opening_core_voxels": int(opening_core.sum()),
            "opening_guard_voxels": int(opening_guard.sum()),
            "known_wrong_fill_protection_voxels": int(known_wrong.sum()),
            "min_opening_retention_2mm": round(min_retention_2mm, 6),
            "min_opening_retention_4mm": round(min_retention_4mm, 6),
            "wall_components_before": component_summary(baseline_wall),
            "wall_components_after": component_summary(final_wall),
        },
        "checksums": {
            "baseline_file_sha256": sha256_bytes(baseline_path),
            "baseline_voxel_array_sha256": sha256_array(baseline_array),
            "output_file_sha256": sha256_bytes(out_path),
            "output_voxel_array_sha256": sha256_array(output_array),
        },
        "geometry": {
            "shape_xyz": pred_vol.shape_xyz,
            "spacing_xyz": pred_vol.spacing_xyz,
            "origin_xyz": pred_vol.origin_xyz,
            "qform_preserved": same_matrix(baseline_vol.qform, pred_vol.qform),
            "sform_preserved": same_matrix(baseline_vol.sform, pred_vol.sform),
        },
        "failures": failures,
    }
    return report
def write_stage2_reports(report_dir: Path, reports: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "biatrial_holes_summary.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "case",
        "stage",
        "stage_id",
        "candidate_voxels",
        "accepted_voxels",
        "RA_exterior_leak_accepted_voxels",
        "LA_exterior_leak_accepted_voxels",
        "new_union_voxels",
        "cumulative_added_voxels",
    ]
    with (report_dir / "biatrial_holes_stages.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for row in report["stages"]:
                writer.writerow({"case": report["case"], **row})
# Standalone runner imports not required by the original algorithm sections.
import platform
import sys
from datetime import datetime, timezone
PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path
def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "patient1.wall_repair_pipeline.v1":
        raise ValueError(f"Unsupported config schema in {path}")
    return payload
def prediction_counts(path: Path) -> dict[str, int]:
    volume = read_nifti(path)
    array = np.rint(volume.array_zyx).astype(np.uint8)
    values, counts = np.unique(array, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}
def wall_result(path: Path) -> dict[str, Any]:
    volume = read_nifti(path)
    array = np.rint(volume.array_zyx).astype(np.uint8)
    labels = sorted(int(value) for value in np.unique(array))
    return {
        "path": str(path),
        "labels": labels,
        "wall_voxels": int(np.count_nonzero(array == WALL_LABEL)),
        "voxel_array_sha256": sha256_array(array),
        "file_sha256": sha256_file(path),
        "shape_xyz": list(volume.shape_xyz),
        "spacing_xyz": list(volume.spacing_xyz),
    }
def preflight(config: dict[str, Any]) -> dict[str, Any]:
    case = str(config["case"])
    paths = config["paths"]
    prediction = resolve_repo_path(paths["prediction"])
    image = resolve_repo_path(paths["image"])
    landmarks = resolve_repo_path(paths["opening_landmarks"])
    wrong_dirs = [resolve_repo_path(path) for path in paths["known_wrong_variant_dirs"]]
    required = [prediction, image, landmarks]
    required.extend(directory / f"{case}.nii.gz" for directory in wrong_dirs)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing pipeline inputs:\n" + "\n".join(missing))
    observed_hashes = {
        "prediction_sha256": sha256_file(prediction),
        "image_sha256": sha256_file(image),
        "opening_landmarks_sha256": sha256_file(landmarks),
        "known_wrong_variant_sha256": {
            directory.name: sha256_file(directory / f"{case}.nii.gz") for directory in wrong_dirs
        },
    }
    expected_hashes = config["input_checksums"]
    if observed_hashes != expected_hashes:
        raise RuntimeError(
            "Input checksum mismatch; refusing to generate a result with unknown provenance.\n"
            + json.dumps({"expected": expected_hashes, "observed": observed_hashes}, indent=2)
        )
    counts = prediction_counts(prediction)
    expected_counts = config["expected_result"]["prediction_label_counts"]
    if counts != expected_counts:
        raise RuntimeError(f"Prediction label counts differ: expected {expected_counts}, observed {counts}")
    prediction_volume = read_nifti(prediction)
    image_volume = read_nifti(image)
    if prediction_volume.shape_xyz != image_volume.shape_xyz:
        raise RuntimeError("Prediction and LGE image shapes differ")
    if not np.allclose(prediction_volume.spacing_xyz, image_volume.spacing_xyz, rtol=0.0, atol=1e-6):
        raise RuntimeError("Prediction and LGE image spacings differ")
    landmark_cases = load_opening_landmarks(landmarks)
    if case not in landmark_cases:
        raise KeyError(f"No reviewed landmark specification for {case}")
    openings = landmark_cases[case].get("openings", [])
    if len(openings) != int(config["stage1_opening_correct_baseline"]["target_opening_count"]):
        raise RuntimeError("Reviewed LA opening count does not match the configured target")
    return {
        "status": "pass",
        "case": case,
        "inputs": {
            "prediction": str(prediction),
            "image": str(image),
            "opening_landmarks": str(landmarks),
            "known_wrong_variant_dirs": [str(path) for path in wrong_dirs],
        },
        "checksums": observed_hashes,
        "prediction_label_counts": counts,
        "geometry": {
            "shape_xyz": list(prediction_volume.shape_xyz),
            "spacing_xyz": list(prediction_volume.spacing_xyz),
        },
        "reviewed_LA_openings": [str(row["opening_class"]) for row in openings],
    }
def verify_outputs(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    case = str(config["case"])
    expected = config["expected_result"]
    baseline_path = run_dir / "01_opening_correct_baseline" / f"{case}.nii.gz"
    final_path = run_dir / "02_biatrial_microholes" / f"{case}.nii.gz"
    summary_path = run_dir / "02_biatrial_microholes_reports" / "biatrial_holes_summary.json"
    for path in (baseline_path, final_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Expected pipeline output is missing: {path}")
    baseline = wall_result(baseline_path)
    final = wall_result(final_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, list) or len(summary) != 1:
        raise RuntimeError(f"Unexpected stage-2 report structure: {summary_path}")
    qc = summary[0]["qc"]
    checks: dict[str, dict[str, Any]] = {}
    def check(name: str, observed: Any, wanted: Any) -> None:
        checks[name] = {"observed": observed, "expected": wanted, "pass": observed == wanted}
    check("baseline_labels", baseline["labels"], [0, 1])
    check("baseline_wall_voxels", baseline["wall_voxels"], expected["baseline_wall_voxels"])
    check(
        "baseline_voxel_array_sha256",
        baseline["voxel_array_sha256"],
        expected["baseline_voxel_array_sha256"],
    )
    check("final_labels", final["labels"], [0, 1])
    check("final_wall_voxels", final["wall_voxels"], expected["final_wall_voxels"])
    check("final_voxel_array_sha256", final["voxel_array_sha256"], expected["final_voxel_array_sha256"])
    check("added_voxels", qc["added_voxels"], expected["added_voxels"])
    check("added_nearer_RA_voxels", qc["added_nearer_RA_voxels"], expected["added_nearer_RA_voxels"])
    check("added_nearer_LA_voxels", qc["added_nearer_LA_voxels"], expected["added_nearer_LA_voxels"])
    check("protected_LA_openings", qc["protected_LA_opening_count"], expected["protected_LA_openings"])
    check("protected_RA_openings", qc["protected_RA_opening_count"], expected["protected_RA_openings"])
    check(
        "minimum_opening_retention_2mm",
        qc["min_opening_retention_2mm"],
        expected["minimum_opening_retention_2mm"],
    )
    check(
        "minimum_opening_retention_4mm",
        qc["min_opening_retention_4mm"],
        expected["minimum_opening_retention_4mm"],
    )
    check("baseline_wall_lost_voxels", qc["baseline_wall_lost_voxels"], 0)
    check("raw_nonbackground_overlap", qc["added_overlap_raw_nonbackground_voxels"], 0)
    check("opening_core_overlap", qc["added_overlap_opening_core_voxels"], 0)
    check("opening_guard_overlap", qc["added_overlap_opening_guard_voxels"], 0)
    check("known_wrong_fill_overlap", qc["added_overlap_known_wrong_fill_voxels"], 0)
    check("wall_component_count", qc["wall_components_after"]["component_count"], 1)
    failures = [name for name, row in checks.items() if not row["pass"]]
    return {
        "status": "pass" if not failures else "fail",
        "case": case,
        "baseline": baseline,
        "final": final,
        "checks": checks,
        "failures": failures,
    }
def run(config: dict[str, Any], run_dir: Path, overwrite: bool) -> dict[str, Any]:
    preflight_report = preflight(config)
    case = str(config["case"])
    paths = config["paths"]
    prediction = resolve_repo_path(paths["prediction"])
    image = resolve_repo_path(paths["image"])
    landmarks_path = resolve_repo_path(paths["opening_landmarks"])
    known_wrong_dirs = [resolve_repo_path(path) for path in paths["known_wrong_variant_dirs"]]
    baseline_dir = run_dir / "01_opening_correct_baseline"
    baseline_report_dir = run_dir / "01_opening_correct_baseline_reports"
    final_dir = run_dir / "02_biatrial_microholes"
    final_report_dir = run_dir / "02_biatrial_microholes_reports"
    expected_outputs = [
        baseline_dir / f"{case}.nii.gz",
        baseline_report_dir / "postprocess_private_pred073_summary.json",
        final_dir / f"{case}.nii.gz",
        final_report_dir / "biatrial_holes_summary.json",
        run_dir / "pipeline_manifest.json",
    ]
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Pipeline outputs already exist. Use a new --run-dir or explicitly pass --overwrite:\n"
            + "\n".join(existing)
        )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_report_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    final_report_dir.mkdir(parents=True, exist_ok=True)
    landmarks = load_opening_landmarks(landmarks_path)
    p1 = config["stage1_opening_correct_baseline"]
    baseline_path = baseline_dir / f"{case}.nii.gz"
    stage1_report = repair_left_atrial_wall(
        pred_path=prediction,
        image_path=image,
        out_path=baseline_path,
        close_mm=float(p1["close_mm"]),
        opening_guard_mm=float(p1["opening_guard_mm"]),
        expected_shell_mm=float(p1["expected_shell_mm"]),
        la_search_radius_mm=float(p1["la_search_radius_mm"]),
        left_wall_radius_mm=float(p1["left_wall_radius_mm"]),
        opening_min_contact_mm3=float(p1["opening_min_contact_mm3"]),
        opening_min_contact_voxels_floor=int(p1["opening_min_contact_voxels_floor"]),
        opening_protect_top_n=int(p1["opening_protect_top_n"]),
        target_opening_count=int(p1["target_opening_count"]),
        opening_selection_mode=str(p1["opening_selection_mode"]),
        pv_max_contact_mm3=float(p1["pv_max_contact_mm3"]),
        pv_min_posterior_rel_y=float(p1["pv_min_posterior_rel_y"]),
        mv_max_rel_z=float(p1["mv_max_rel_z"]),
        output_mode=str(p1["output_mode"]),
        wall_min_component_voxels=int(p1["wall_min_component_voxels"]),
        cavity_min_component_voxels=int(p1["cavity_min_component_voxels"]),
        max_internal_hole_voxels=int(p1["max_internal_hole_voxels"]),
        reviewed_landmark_spec=landmarks[case],
    )
    write_stage1_reports(baseline_report_dir, [stage1_report])
    if stage1_report["status"] != "pass":
        raise RuntimeError(f"Stage 1 did not pass QC: {stage1_report['failures']}")
    baseline_report_path = baseline_report_dir / "postprocess_private_pred073_summary.json"
    p2 = config["stage2_biatrial_microholes"]
    stage2_report = repair_case(
        case=case,
        pred_dir=prediction.parent,
        image_dir=image.parent,
        baseline_dir=baseline_dir,
        baseline_report_path=baseline_report_path,
        baseline_report=stage1_report,
        known_wrong_dirs=known_wrong_dirs,
        output_dir=final_dir,
        report_dir=final_report_dir,
        passes=int(p2["passes"]),
        shell_mm=float(p2["shell_mm"]),
        opening_guard_mm=float(p2["opening_guard_mm"]),
        wall_search_mm=float(p2["wall_search_mm"]),
        max_hole_component_voxels=int(p2["max_hole_component_voxels"]),
        ra_opening_count=int(p2["ra_opening_count"]),
        ra_opening_min_contact_voxels=int(p2["ra_opening_min_contact_voxels"]),
        max_total_additions=int(p2["max_total_additions"]),
        overwrite=overwrite,
    )
    write_stage2_reports(final_report_dir, [stage2_report])
    verification = verify_outputs(config, run_dir)
    manifest = {
        "schema_version": "patient1.wall_repair_run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": verification["status"],
        "case": case,
        "config": str(PIPELINE_DIR / "config.json"),
        "run_dir": str(run_dir),
        "preflight": preflight_report,
        "code": {
            "standalone_pipeline": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            }
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "outputs": {
            "baseline": str(baseline_path),
            "baseline_report": str(baseline_report_path),
            "final": str(final_dir / f"{case}.nii.gz"),
            "final_report": str(final_report_dir / "biatrial_holes_summary.json"),
            "qc_montage": str(final_report_dir / f"{case}_biatrial_holes_qc.png"),
        },
        "verification": verification,
    }
    manifest_path = run_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if verification["status"] != "pass":
        raise RuntimeError(f"Final verification failed: {verification['failures']}")
    return manifest
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the reviewed patient1 bilateral atrial-wall result from the preserved prediction."
    )
    parser.add_argument("--config", type=Path, default=PIPELINE_DIR / "config.json")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    run_dir = (args.run_dir or resolve_repo_path(config["paths"]["default_run_dir"])).resolve()
    if args.preflight_only:
        report = preflight(config)
    elif args.verify_only:
        report = verify_outputs(config, run_dir)
        if report["status"] != "pass":
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 1
    else:
        report = run(config, run_dir, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
