from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from skimage import measure


REPRO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPRO / "private_pred073_postprocessed_close7"
DEFAULT_OUTPUT_DIR = REPRO / "meshes_private_pred073_close7"
SCHEMA_VERSION = "mesh_private_pred073.pipeline.v2"

DEFAULT_LABELS = {
    1: "biatrial_wall",
    2: "RA",
    3: "LA",
}

STATUS_RANK = {
    "pass": 0,
    "warning": 1,
    "fail": 2,
}


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class QaThresholds:
    min_vertices: int = 10
    min_faces: int = 10
    max_boundary_edges: int = 0
    max_nonmanifold_edges: int = 0
    max_components_warning: int = 1
    max_components_fail: int = 3
    min_largest_component_fraction_warning: float = 0.99
    min_largest_component_fraction_fail: float = 0.95
    max_secondary_component_faces_warning: int = 50
    max_secondary_component_faces_fail: int = 500
    max_area_change_fraction_warning: float = 0.05
    max_area_change_fraction_fail: float = 0.15

    def update(self, values: dict[str, Any]) -> None:
        valid = set(asdict(self))
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(f"Unknown QA threshold keys: {unknown}")
        for key, value in values.items():
            setattr(self, key, value)


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def list_nifti_files(directory: Path) -> list[Path]:
    return sorted([*directory.glob("*.nii.gz"), *directory.glob("*.nii")])


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def load_labels(input_dir: Path) -> dict[int, str]:
    dataset_json = input_dir / "dataset.json"
    if not dataset_json.exists():
        return DEFAULT_LABELS

    with dataset_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    labels: dict[int, str] = {}
    for name, value in data.get("labels", {}).items():
        if name.lower() == "background":
            continue
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        try:
            label_id = int(value)
        except (TypeError, ValueError):
            continue
        if label_id > 0:
            labels[label_id] = name.replace("biatrail", "biatrial")

    return dict(sorted(labels.items())) if labels else DEFAULT_LABELS


def parse_labels(text: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    for item in text.split(","):
        label_id, label_name = item.split(":", 1)
        labels[int(label_id)] = label_name
    return dict(sorted(labels.items()))


def load_segmentation(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    seg = np.asanyarray(img.dataobj)
    seg = np.rint(seg).astype(np.int16)
    if seg.ndim != 3:
        raise ValueError(f"{path}: expected a 3D label map, got shape={seg.shape}")
    return img, seg


def extract_surface(mask: np.ndarray, affine: np.ndarray, step_size: int = 1) -> Mesh:
    if not mask.any():
        return Mesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
        )

    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    vertices_ijk, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
        step_size=step_size,
        allow_degenerate=False,
    )
    vertices_ijk -= 1.0
    vertices_xyz = nib.affines.apply_affine(affine, vertices_ijk)
    return Mesh(vertices=vertices_xyz.astype(np.float64), faces=faces.astype(np.int64))


def build_vertex_neighbors(faces: np.ndarray, vertex_count: int) -> list[set[int]]:
    neighbors = [set() for _ in range(vertex_count)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def laplacian_smooth(mesh: Mesh, iterations: int, relaxation: float) -> Mesh:
    if iterations <= 0 or mesh.vertices.size == 0:
        return mesh

    vertices = mesh.vertices.copy()
    neighbors = build_vertex_neighbors(mesh.faces, len(vertices))
    for _ in range(iterations):
        updated = vertices.copy()
        for index, linked in enumerate(neighbors):
            if linked:
                mean_neighbor = vertices[list(linked)].mean(axis=0)
                updated[index] = vertices[index] + relaxation * (mean_neighbor - vertices[index])
        vertices = updated
    return Mesh(vertices=vertices, faces=mesh.faces)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def component_ids(mesh: Mesh) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    if mesh.faces.size == 0:
        return np.zeros(0, dtype=np.int64), {}

    dsu = DisjointSet(len(mesh.vertices))
    for a, b, c in mesh.faces:
        dsu.union(int(a), int(b))
        dsu.union(int(a), int(c))

    face_roots = np.array([dsu.find(int(face[0])) for face in mesh.faces], dtype=np.int64)
    face_counts = Counter(face_roots.tolist())

    vertex_counts: Counter[int] = Counter()
    used_vertices = np.unique(mesh.faces.ravel())
    for vertex in used_vertices:
        vertex_counts[dsu.find(int(vertex))] += 1

    stats = {
        int(root): {
            "vertices": int(vertex_counts[root]),
            "faces": int(face_counts[root]),
        }
        for root in sorted(face_counts, key=lambda root: face_counts[root], reverse=True)
    }
    return face_roots, stats


def filter_components(mesh: Mesh, min_component_faces: int, keep_largest_only: bool) -> Mesh:
    if mesh.faces.size == 0 or (min_component_faces <= 0 and not keep_largest_only):
        return mesh

    face_roots, stats = component_ids(mesh)
    if not stats:
        return mesh

    largest_root = max(stats, key=lambda root: stats[root]["faces"])
    keep_roots = set()
    for root, item in stats.items():
        if keep_largest_only and root != largest_root:
            continue
        if item["faces"] >= min_component_faces or root == largest_root:
            keep_roots.add(root)

    keep_faces = np.isin(face_roots, list(keep_roots))
    faces = mesh.faces[keep_faces]
    used_vertices = np.unique(faces.ravel())
    index_map = np.full(len(mesh.vertices), -1, dtype=np.int64)
    index_map[used_vertices] = np.arange(len(used_vertices))
    return Mesh(vertices=mesh.vertices[used_vertices], faces=index_map[faces])


def edge_counts(faces: np.ndarray) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for a, b, c in faces:
        for left, right in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(left), int(right))))
            counts[edge] += 1
    return counts


def surface_area_mm2(mesh: Mesh) -> float:
    if mesh.faces.size == 0:
        return 0.0
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def mesh_stats(mesh: Mesh) -> dict[str, Any]:
    _, components = component_ids(mesh)
    edges = edge_counts(mesh.faces)
    component_items = list(components.values())
    largest_faces = max((item["faces"] for item in component_items), default=0)
    sorted_component_faces = sorted((item["faces"] for item in component_items), reverse=True)
    second_largest_faces = sorted_component_faces[1] if len(sorted_component_faces) > 1 else 0

    if mesh.vertices.size == 0:
        bounds = {"min_xyz_mm": [None, None, None], "max_xyz_mm": [None, None, None]}
    else:
        bounds = {
            "min_xyz_mm": mesh.vertices.min(axis=0).round(6).tolist(),
            "max_xyz_mm": mesh.vertices.max(axis=0).round(6).tolist(),
        }

    face_count = int(len(mesh.faces))
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": face_count,
        "surface_area_mm2": surface_area_mm2(mesh),
        "component_count": int(len(component_items)),
        "largest_component_faces": int(largest_faces),
        "largest_component_face_fraction": float(largest_faces / face_count) if face_count else 0.0,
        "second_largest_component_faces": int(second_largest_faces),
        "edge_count": int(len(edges)),
        "boundary_edges": int(sum(1 for count in edges.values() if count == 1)),
        "nonmanifold_edges": int(sum(1 for count in edges.values() if count > 2)),
        "bounds": bounds,
        "components": component_items,
    }


def make_issue(
    level: str,
    code: str,
    message: str,
    metric: str | None = None,
    value: Any | None = None,
    threshold: Any | None = None,
) -> dict[str, Any]:
    issue = {
        "level": level,
        "code": code,
        "message": message,
    }
    if metric is not None:
        issue["metric"] = metric
    if value is not None:
        issue["value"] = value
    if threshold is not None:
        issue["threshold"] = threshold
    return issue


def status_from_issues(issues: list[dict[str, Any]]) -> str:
    status = "pass"
    for issue in issues:
        level = issue["level"]
        if STATUS_RANK[level] > STATUS_RANK[status]:
            status = level
    return status


def evaluate_surface_qa(
    surface_name: str,
    raw_stats: dict[str, Any],
    final_stats: dict[str, Any],
    thresholds: QaThresholds,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    if final_stats["vertices"] < thresholds.min_vertices:
        issues.append(
            make_issue(
                "fail",
                "too_few_vertices",
                f"{surface_name}: surface has too few vertices",
                "vertices",
                final_stats["vertices"],
                thresholds.min_vertices,
            )
        )

    if final_stats["faces"] < thresholds.min_faces:
        issues.append(
            make_issue(
                "fail",
                "too_few_faces",
                f"{surface_name}: surface has too few faces",
                "faces",
                final_stats["faces"],
                thresholds.min_faces,
            )
        )

    if final_stats["boundary_edges"] > thresholds.max_boundary_edges:
        issues.append(
            make_issue(
                "fail",
                "boundary_edges",
                f"{surface_name}: unexpected open boundary edges were detected",
                "boundary_edges",
                final_stats["boundary_edges"],
                thresholds.max_boundary_edges,
            )
        )

    if final_stats["nonmanifold_edges"] > thresholds.max_nonmanifold_edges:
        issues.append(
            make_issue(
                "fail",
                "nonmanifold_edges",
                f"{surface_name}: non-manifold edges were detected",
                "nonmanifold_edges",
                final_stats["nonmanifold_edges"],
                thresholds.max_nonmanifold_edges,
            )
        )

    if final_stats["component_count"] > thresholds.max_components_fail:
        issues.append(
            make_issue(
                "fail",
                "too_many_components",
                f"{surface_name}: too many disconnected mesh components",
                "component_count",
                final_stats["component_count"],
                thresholds.max_components_fail,
            )
        )
    elif final_stats["component_count"] > thresholds.max_components_warning:
        issues.append(
            make_issue(
                "warning",
                "multiple_components",
                f"{surface_name}: more than one disconnected mesh component",
                "component_count",
                final_stats["component_count"],
                thresholds.max_components_warning,
            )
        )

    fraction = final_stats["largest_component_face_fraction"]
    if fraction < thresholds.min_largest_component_fraction_fail:
        issues.append(
            make_issue(
                "fail",
                "low_largest_component_fraction",
                f"{surface_name}: largest component is too small relative to the full surface",
                "largest_component_face_fraction",
                fraction,
                thresholds.min_largest_component_fraction_fail,
            )
        )
    elif fraction < thresholds.min_largest_component_fraction_warning:
        issues.append(
            make_issue(
                "warning",
                "low_largest_component_fraction",
                f"{surface_name}: secondary components are present",
                "largest_component_face_fraction",
                fraction,
                thresholds.min_largest_component_fraction_warning,
            )
        )

    second_largest = final_stats["second_largest_component_faces"]
    if second_largest > thresholds.max_secondary_component_faces_fail:
        issues.append(
            make_issue(
                "fail",
                "large_secondary_component",
                f"{surface_name}: a large secondary component remains",
                "second_largest_component_faces",
                second_largest,
                thresholds.max_secondary_component_faces_fail,
            )
        )
    elif second_largest > thresholds.max_secondary_component_faces_warning:
        issues.append(
            make_issue(
                "warning",
                "secondary_component",
                f"{surface_name}: a secondary component remains",
                "second_largest_component_faces",
                second_largest,
                thresholds.max_secondary_component_faces_warning,
            )
        )

    raw_area = raw_stats["surface_area_mm2"]
    final_area = final_stats["surface_area_mm2"]
    if raw_area > 0:
        area_change_fraction = abs(final_area - raw_area) / raw_area
        if area_change_fraction > thresholds.max_area_change_fraction_fail:
            issues.append(
                make_issue(
                    "fail",
                    "surface_area_changed",
                    f"{surface_name}: mesh cleanup changed surface area too much",
                    "surface_area_change_fraction",
                    area_change_fraction,
                    thresholds.max_area_change_fraction_fail,
                )
            )
        elif area_change_fraction > thresholds.max_area_change_fraction_warning:
            issues.append(
                make_issue(
                    "warning",
                    "surface_area_changed",
                    f"{surface_name}: mesh cleanup changed surface area noticeably",
                    "surface_area_change_fraction",
                    area_change_fraction,
                    thresholds.max_area_change_fraction_warning,
                )
            )

    return {
        "status": status_from_issues(issues),
        "issues": issues,
    }


def write_obj(path: Path, mesh: Mesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {path.name}\n")
        for x, y, z in mesh.vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in mesh.faces:
            f.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")


def write_vtk(path: Path, mesh: Mesh, cell_labels: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = cell_labels if cell_labels is not None else np.ones(len(mesh.faces), dtype=np.int16)

    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{path.stem}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(mesh.vertices)} float\n")
        for x, y, z in mesh.vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        f.write(f"POLYGONS {len(mesh.faces)} {len(mesh.faces) * 4}\n")
        for a, b, c in mesh.faces:
            f.write(f"3 {int(a)} {int(b)} {int(c)}\n")
        f.write(f"CELL_DATA {len(mesh.faces)}\n")
        f.write("SCALARS label_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for label in labels:
            f.write(f"{int(label)}\n")


def combine_meshes(mesh_items: list[tuple[int, Mesh]]) -> tuple[Mesh, np.ndarray]:
    vertices = []
    faces = []
    labels = []
    offset = 0
    for label_id, mesh in mesh_items:
        if mesh.faces.size == 0:
            continue
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        labels.extend([label_id] * len(mesh.faces))
        offset += len(mesh.vertices)

    if not vertices:
        return Mesh(np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)), np.zeros(0)

    return Mesh(np.vstack(vertices), np.vstack(faces)), np.asarray(labels, dtype=np.int16)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_surface_record(
    label_id: int | str,
    label_name: str,
    mask: np.ndarray,
    affine: np.ndarray,
    case_dir: Path,
    step_size: int,
    smooth_iters: int,
    smooth_relaxation: float,
    min_component_faces: int,
    keep_largest_only: bool,
    thresholds: QaThresholds,
) -> tuple[Mesh, dict[str, Any]]:
    raw_mesh = extract_surface(mask, affine=affine, step_size=step_size)
    raw_stats = mesh_stats(raw_mesh)

    final_mesh = laplacian_smooth(raw_mesh, iterations=smooth_iters, relaxation=smooth_relaxation)
    final_mesh = filter_components(final_mesh, min_component_faces, keep_largest_only)
    final_stats = mesh_stats(final_mesh)

    qa = evaluate_surface_qa(label_name, raw_stats, final_stats, thresholds)

    stem = f"{label_id}_{safe_name(label_name)}" if isinstance(label_id, int) else safe_name(label_name)
    obj_path = case_dir / f"{stem}.obj"
    vtk_path = case_dir / f"{stem}.vtk"
    vtk_label = label_id if isinstance(label_id, int) else 0

    write_obj(obj_path, final_mesh)
    write_vtk(vtk_path, final_mesh, np.full(len(final_mesh.faces), vtk_label, dtype=np.int16))

    record = {
        "label_id": label_id,
        "label_name": label_name,
        "voxel_count": int(mask.sum()),
        "outputs": {
            "obj": str(obj_path),
            "vtk": str(vtk_path),
        },
        "raw_surface": raw_stats,
        "final_surface": final_stats,
        "qa": qa,
    }
    return final_mesh, record


def process_case(
    nii_path: Path,
    output_dir: Path,
    labels: dict[int, str],
    smooth_iters: int,
    smooth_relaxation: float,
    step_size: int,
    min_component_faces: int,
    keep_largest_only: bool,
    include_foreground: bool,
    thresholds: QaThresholds,
) -> dict[str, Any]:
    img, seg = load_segmentation(nii_path)
    case = strip_nii_suffix(nii_path)
    case_dir = output_dir / case
    affine = img.affine
    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])

    case_record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case": case,
        "input": str(nii_path),
        "spacing": spacing,
        "output_dir": str(case_dir),
        "status": "pass",
        "issues": [],
        "surfaces": {},
        "outputs": {},
    }

    combined_items: list[tuple[int, Mesh]] = []
    for label_id, label_name in labels.items():
        final_mesh, surface_record = build_surface_record(
            label_id=label_id,
            label_name=label_name,
            mask=seg == label_id,
            affine=affine,
            case_dir=case_dir,
            step_size=step_size,
            smooth_iters=smooth_iters,
            smooth_relaxation=smooth_relaxation,
            min_component_faces=min_component_faces,
            keep_largest_only=keep_largest_only,
            thresholds=thresholds,
        )
        combined_items.append((label_id, final_mesh))
        case_record["surfaces"][label_name] = surface_record

        for issue in surface_record["qa"]["issues"]:
            case_record["issues"].append({"surface": label_name, **issue})

    if include_foreground:
        _, foreground_record = build_surface_record(
            label_id="nonzero",
            label_name="foreground",
            mask=seg > 0,
            affine=affine,
            case_dir=case_dir,
            step_size=step_size,
            smooth_iters=smooth_iters,
            smooth_relaxation=smooth_relaxation,
            min_component_faces=min_component_faces,
            keep_largest_only=keep_largest_only,
            thresholds=thresholds,
        )
        case_record["surfaces"]["foreground"] = foreground_record
        for issue in foreground_record["qa"]["issues"]:
            case_record["issues"].append({"surface": "foreground", **issue})

    combined_mesh, combined_labels = combine_meshes(combined_items)
    combined_path = case_dir / "labels_combined.vtk"
    write_vtk(combined_path, combined_mesh, combined_labels)
    case_record["outputs"]["combined_vtk"] = str(combined_path)

    case_record["status"] = status_from_issues(case_record["issues"])
    case_qc_path = case_dir / "case_qc.json"
    case_record["outputs"]["case_qc"] = str(case_qc_path)
    write_json(case_qc_path, case_record)

    return case_record


def load_thresholds(args: argparse.Namespace) -> QaThresholds:
    thresholds = QaThresholds()

    if args.qa_thresholds is not None:
        with args.qa_thresholds.open("r", encoding="utf-8") as f:
            thresholds.update(json.load(f))

    overrides = {
        "min_vertices": args.qa_min_vertices,
        "min_faces": args.qa_min_faces,
        "max_boundary_edges": args.qa_max_boundary_edges,
        "max_nonmanifold_edges": args.qa_max_nonmanifold_edges,
        "max_components_warning": args.qa_max_components_warning,
        "max_components_fail": args.qa_max_components_fail,
        "min_largest_component_fraction_warning": args.qa_min_largest_component_fraction_warning,
        "min_largest_component_fraction_fail": args.qa_min_largest_component_fraction_fail,
        "max_secondary_component_faces_warning": args.qa_max_secondary_component_faces_warning,
        "max_secondary_component_faces_fail": args.qa_max_secondary_component_faces_fail,
        "max_area_change_fraction_warning": args.qa_max_area_change_fraction_warning,
        "max_area_change_fraction_fail": args.qa_max_area_change_fraction_fail,
    }
    thresholds.update({key: value for key, value in overrides.items() if value is not None})
    return thresholds


def build_summary(
    input_dir: Path,
    output_dir: Path,
    labels: dict[int, str],
    thresholds: QaThresholds,
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(case["status"] for case in cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "labels": {str(label_id): label_name for label_id, label_name in labels.items()},
        "qa_thresholds": asdict(thresholds),
        "parameters": {
            "step_size": args.step_size,
            "smooth_iters": args.smooth_iters,
            "smooth_relaxation": args.smooth_relaxation,
            "min_component_faces": args.min_component_faces,
            "keep_largest_only": args.keep_largest_only,
            "include_foreground": not args.no_foreground,
        },
        "aggregate": {
            "case_count": len(cases),
            "status_counts": {status: int(status_counts.get(status, 0)) for status in STATUS_RANK},
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert postprocessed 3-label atrial NIfTI segmentations to QA-gated surface meshes."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", type=str, default=None, help="Example: 1:biatrial_wall,2:RA,3:LA")
    parser.add_argument("--smooth-iters", type=int, default=0)
    parser.add_argument("--smooth-relaxation", type=float, default=0.2)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--min-component-faces", type=int, default=0)
    parser.add_argument("--keep-largest-only", action="store_true")
    parser.add_argument("--no-foreground", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--qa-thresholds", type=Path, default=None)
    parser.add_argument("--qa-min-vertices", type=int, default=None)
    parser.add_argument("--qa-min-faces", type=int, default=None)
    parser.add_argument("--qa-max-boundary-edges", type=int, default=None)
    parser.add_argument("--qa-max-nonmanifold-edges", type=int, default=None)
    parser.add_argument("--qa-max-components-warning", type=int, default=None)
    parser.add_argument("--qa-max-components-fail", type=int, default=None)
    parser.add_argument("--qa-min-largest-component-fraction-warning", type=float, default=None)
    parser.add_argument("--qa-min-largest-component-fraction-fail", type=float, default=None)
    parser.add_argument("--qa-max-secondary-component-faces-warning", type=int, default=None)
    parser.add_argument("--qa-max-secondary-component-faces-fail", type=int, default=None)
    parser.add_argument("--qa-max-area-change-fraction-warning", type=float, default=None)
    parser.add_argument("--qa-max-area-change-fraction-fail", type=float, default=None)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists and is not empty; use --overwrite")

    labels = parse_labels(args.labels) if args.labels else load_labels(args.input_dir)
    thresholds = load_thresholds(args)
    nii_files = list_nifti_files(args.input_dir)
    if not nii_files:
        raise FileNotFoundError(f"No NIfTI files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for nii_path in nii_files:
        case_record = process_case(
            nii_path=nii_path,
            output_dir=args.output_dir,
            labels=labels,
            smooth_iters=args.smooth_iters,
            smooth_relaxation=args.smooth_relaxation,
            step_size=args.step_size,
            min_component_faces=args.min_component_faces,
            keep_largest_only=args.keep_largest_only,
            include_foreground=not args.no_foreground,
            thresholds=thresholds,
        )
        cases.append(case_record)
        print(f"[{case_record['status'].upper()}] {nii_path.name}")

    summary = build_summary(args.input_dir, args.output_dir, labels, thresholds, args, cases)
    summary_path = args.output_dir / "mesh_summary.json"
    write_json(summary_path, summary)
    print(f"[SAVE] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
