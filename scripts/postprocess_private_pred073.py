from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


REPRO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPRO / "private_pred073"
DEFAULT_OUTPUT_DIR = REPRO / "private_pred073_postprocessed"

LABELS = {
    1: "biatrial_wall",
    2: "RA",
    3: "LA",
}


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def list_nifti_files(directory: Path) -> list[Path]:
    return sorted([*directory.glob("*.nii.gz"), *directory.glob("*.nii")])


def physical_ball(spacing: tuple[float, float, float], radius_mm: float) -> np.ndarray:
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)

    spacing_arr = np.asarray(spacing, dtype=np.float64)
    if np.any(spacing_arr <= 0):
        raise ValueError(f"Invalid voxel spacing: {spacing}")

    radii = [int(np.ceil(radius_mm / s)) for s in spacing_arr]
    grids = np.meshgrid(*[np.arange(-r, r + 1) for r in radii], indexing="ij")
    dist2 = sum((grid * s) ** 2 for grid, s in zip(grids, spacing_arr))
    return dist2 <= radius_mm**2


def keep_largest_with_large_islands(
    mask: np.ndarray,
    conn: int = 1,
    min_voxels: int = 500,
) -> tuple[np.ndarray, list[dict]]:
    structure = ndi.generate_binary_structure(mask.ndim, conn)
    cc, num_components = ndi.label(mask, structure=structure)

    if num_components == 0:
        return mask, []

    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))

    cleaned = np.zeros_like(mask, dtype=bool)
    components = []

    for component_id, size in enumerate(sizes):
        if component_id == 0:
            continue

        keep = component_id == largest or size >= min_voxels
        if keep:
            cleaned |= cc == component_id

        components.append(
            {
                "component_id": int(component_id),
                "voxels": int(size),
                "kept": bool(keep),
                "largest": bool(component_id == largest),
            }
        )

    return cleaned, components


def border_component_ids(cc: np.ndarray) -> set[int]:
    labels = np.concatenate(
        [
            cc[0, :, :].ravel(),
            cc[-1, :, :].ravel(),
            cc[:, 0, :].ravel(),
            cc[:, -1, :].ravel(),
            cc[:, :, 0].ravel(),
            cc[:, :, -1].ravel(),
        ]
    )
    ids = set(np.unique(labels).astype(int).tolist())
    ids.discard(0)
    return ids


def neighbor_label_counts(component_mask: np.ndarray, seg: np.ndarray) -> dict[int, int]:
    structure = ndi.generate_binary_structure(component_mask.ndim, 1)
    shell = ndi.binary_dilation(component_mask, structure=structure) & ~component_mask
    values = seg[shell]
    values = values[values > 0]

    if values.size == 0:
        return {}

    counts = np.bincount(values.astype(np.int16), minlength=max(LABELS) + 1)
    return {label_id: int(counts[label_id]) for label_id in LABELS if counts[label_id] > 0}


def choose_fill_label(counts: dict[int, int]) -> int | None:
    if not counts:
        return None

    # A background hole that touches the wall is treated as a wall surface defect.
    if counts.get(1, 0) > 0:
        return 1

    # If the hole is fully inside RA or LA, fill it with the dominant cavity label.
    return max(counts, key=counts.get)


def fill_internal_background_holes(
    seg: np.ndarray,
    max_hole_voxels: int = 5000,
    conn: int = 1,
) -> tuple[np.ndarray, list[dict]]:
    background = seg == 0
    structure = ndi.generate_binary_structure(seg.ndim, conn)
    cc, num_components = ndi.label(background, structure=structure)

    if num_components == 0:
        return seg, []

    border_ids = border_component_ids(cc)
    sizes = np.bincount(cc.ravel())

    out = seg.copy()
    filled = []

    for component_id in range(1, num_components + 1):
        if component_id in border_ids:
            continue

        size = int(sizes[component_id])
        if size > max_hole_voxels:
            continue

        component = cc == component_id
        counts = neighbor_label_counts(component, out)
        fill_label = choose_fill_label(counts)
        if fill_label is None:
            continue

        out[component] = fill_label
        filled.append(
            {
                "component_id": int(component_id),
                "voxels": size,
                "fill_label": int(fill_label),
                "neighbor_counts": counts,
            }
        )

    return out, filled


def repair_wall_surface_pits(
    seg: np.ndarray,
    spacing: tuple[float, float, float],
    close_mm: float = 1.8,
    max_patch_voxels: int = 5000,
    conn: int = 1,
) -> tuple[np.ndarray, list[dict]]:
    wall = seg == 1
    if not wall.any():
        return seg, []

    structure = physical_ball(spacing, close_mm)
    closed_wall = ndi.binary_closing(wall, structure=structure)
    candidates = closed_wall & (seg == 0)

    if not candidates.any():
        return seg, []

    near_wall = ndi.distance_transform_edt(~wall, sampling=spacing) <= close_mm
    candidates &= near_wall

    cc, num_components = ndi.label(
        candidates,
        structure=ndi.generate_binary_structure(seg.ndim, conn),
    )
    if num_components == 0:
        return seg, []

    sizes = np.bincount(cc.ravel())
    out = seg.copy()
    patches = []

    for component_id in range(1, num_components + 1):
        size = int(sizes[component_id])
        if size > max_patch_voxels:
            continue

        component = cc == component_id
        counts = neighbor_label_counts(component, out)
        if counts.get(1, 0) == 0:
            continue

        out[component] = 1
        patches.append(
            {
                "component_id": int(component_id),
                "voxels": size,
                "neighbor_counts": counts,
            }
        )

    return out, patches


def load_segmentation(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    seg = np.asanyarray(img.dataobj)
    seg = np.rint(seg).astype(np.uint8)

    if seg.ndim != 3:
        raise ValueError(f"{path}: expected 3D label map, got shape={seg.shape}")

    valid_labels = {0, *LABELS.keys()}
    present_labels = set(np.unique(seg).astype(int).tolist())
    unknown_labels = sorted(present_labels - valid_labels)
    if unknown_labels:
        raise ValueError(f"{path}: unexpected labels {unknown_labels}; expected {sorted(valid_labels)}")

    return img, seg


def save_segmentation(path: Path, img: nib.Nifti1Image, seg: np.ndarray) -> None:
    header = img.header.copy()
    header.set_data_dtype(np.uint8)

    out = nib.Nifti1Image(seg.astype(np.uint8), img.affine, header)
    out.set_qform(img.get_qform(), code=int(img.header["qform_code"]))
    out.set_sform(img.get_sform(), code=int(img.header["sform_code"]))
    out.set_data_dtype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))


def postprocess_label(
    in_nii: Path,
    out_nii: Path,
    close_mm: float = 1.8,
    wall_min_component_voxels: int = 500,
    cavity_min_component_voxels: int = 500,
    max_internal_hole_voxels: int = 5000,
    max_wall_patch_voxels: int = 5000,
    conn: int = 1,
) -> dict:
    img, seg = load_segmentation(in_nii)
    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])

    clean = np.zeros_like(seg, dtype=np.uint8)
    component_stats = {}

    # Step 1: remove small disconnected islands for each label.
    for label_id, label_name in LABELS.items():
        min_voxels = wall_min_component_voxels if label_id == 1 else cavity_min_component_voxels
        label_mask, label_components = keep_largest_with_large_islands(
            seg == label_id,
            conn=conn,
            min_voxels=min_voxels,
        )
        clean[label_mask] = label_id
        component_stats[label_name] = label_components

    # Step 2: fill small enclosed background holes. Holes touching wall are filled as wall.
    clean, filled_internal_holes = fill_internal_background_holes(
        clean,
        max_hole_voxels=max_internal_hole_voxels,
        conn=conn,
    )

    # Step 3: close small wall surface pits that are not fully enclosed background holes.
    clean, filled_wall_patches = repair_wall_surface_pits(
        clean,
        spacing=spacing,
        close_mm=close_mm,
        max_patch_voxels=max_wall_patch_voxels,
        conn=conn,
    )

    save_segmentation(out_nii, img, clean)

    stats = {
        "case": strip_nii_suffix(in_nii),
        "input": str(in_nii),
        "output": str(out_nii),
        "spacing": spacing,
        "component_cleanup": component_stats,
        "filled_internal_background_holes": filled_internal_holes,
        "filled_wall_surface_patches": filled_wall_patches,
        "labels": {},
    }
    for label_id, label_name in LABELS.items():
        stats["labels"][label_name] = {
            "before_voxels": int((seg == label_id).sum()),
            "after_voxels": int((clean == label_id).sum()),
        }

    return stats


def copy_sidecar_files(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.iterdir():
        if path.suffix.lower() == ".json":
            shutil.copy2(path, output_dir / path.name)


def write_summary(output_dir: Path, stats: list[dict]) -> None:
    summary_path = output_dir / "postprocess_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVE] {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--close-mm", type=float, default=1.8)
    parser.add_argument("--wall-min-component-voxels", type=int, default=500)
    parser.add_argument("--cavity-min-component-voxels", type=int, default=500)
    parser.add_argument("--max-internal-hole-voxels", type=int, default=5000)
    parser.add_argument("--max-wall-patch-voxels", type=int, default=5000)
    parser.add_argument("--conn", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecar_files(args.input_dir, args.output_dir)

    nii_files = list_nifti_files(args.input_dir)
    print(f"INPUT_DIR={args.input_dir}")
    print(f"OUTPUT_DIR={args.output_dir}")
    print(f"num_cases={len(nii_files)}")
    print(f"labels={LABELS}")

    all_stats = []
    for in_nii in nii_files:
        out_nii = args.output_dir / in_nii.name
        if out_nii.exists() and not args.overwrite:
            print(f"[SKIP] {in_nii.name}: output exists, use --overwrite to regenerate")
            continue

        stats = postprocess_label(
            in_nii=in_nii,
            out_nii=out_nii,
            close_mm=args.close_mm,
            wall_min_component_voxels=args.wall_min_component_voxels,
            cavity_min_component_voxels=args.cavity_min_component_voxels,
            max_internal_hole_voxels=args.max_internal_hole_voxels,
            max_wall_patch_voxels=args.max_wall_patch_voxels,
            conn=args.conn,
        )
        all_stats.append(stats)
        print(f"[OK] {in_nii.name} -> {out_nii}")

    write_summary(args.output_dir, all_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
