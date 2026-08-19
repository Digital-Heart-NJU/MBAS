from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


REPRO = Path(__file__).resolve().parents[1]
DEFAULT_GT_DIR = REPRO / "gtTs073_mapped"
DEFAULT_PRED_DIR = REPRO / "predTs073"
DEFAULT_OUT_DIR = REPRO / "logs"

DEFAULT_LABELS = {
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


def parse_labels(text: str) -> dict[int, str]:
    labels = {}
    for item in text.split(","):
        label_id, label_name = item.split(":", 1)
        labels[int(label_id)] = label_name
    return dict(sorted(labels.items()))


def load_labels(pred_dir: Path) -> dict[int, str]:
    dataset_json = pred_dir / "dataset.json"
    if not dataset_json.exists():
        return DEFAULT_LABELS

    with dataset_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    labels = {}
    for name, value in data.get("labels", {}).items():
        if name.lower() == "background":
            continue

        if isinstance(value, list) and len(value) == 1:
            value = value[0]

        try:
            label_id = int(value)
        except (TypeError, ValueError):
            continue

        if label_id != 0:
            labels[label_id] = name.replace("biatrail", "biatrial")

    return dict(sorted(labels.items())) if labels else DEFAULT_LABELS


def find_pred(case: str, pred_dir: Path) -> Path | None:
    stems = [
        case,
        f"{case}_label",
        f"{case}_seg",
        f"{case}_prediction",
    ]

    for stem in stems:
        for suffix in (".nii.gz", ".nii"):
            path = pred_dir / f"{stem}{suffix}"
            if path.exists():
                return path

    matches = sorted(pred_dir.glob(f"*{case}*.nii.gz")) + sorted(pred_dir.glob(f"*{case}*.nii"))
    return matches[0] if matches else None


def load_label_array(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    arr = np.rint(arr).astype(np.int16)
    return img, arr


def dice_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    gt_sum = int(gt_mask.sum())
    pred_sum = int(pred_mask.sum())

    if gt_sum == 0 and pred_sum == 0:
        return 1.0
    if gt_sum == 0 or pred_sum == 0:
        return 0.0

    intersection = int(np.logical_and(gt_mask, pred_mask).sum())
    return float(2.0 * intersection / (gt_sum + pred_sum))


def surface_mask(mask: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def hd95_score(gt_mask: np.ndarray, pred_mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    if not gt_mask.any() and not pred_mask.any():
        return 0.0
    if not gt_mask.any() or not pred_mask.any():
        return float("inf")

    gt_surface = surface_mask(gt_mask)
    pred_surface = surface_mask(pred_mask)

    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    pred_to_gt = dt_gt[pred_surface]
    del dt_gt

    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    gt_to_pred = dt_pred[gt_surface]
    del dt_pred

    distances = np.concatenate([pred_to_gt, gt_to_pred])
    if distances.size == 0:
        return 0.0

    return float(np.percentile(distances, 95))


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def format_metric(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict], labels: dict[int, str]) -> list[dict]:
    summary = []

    for label_id, label_name in labels.items():
        label_rows = [r for r in rows if r["label_id"] == label_id]
        summary.append(
            {
                "scope": "label",
                "case": "ALL",
                "label_id": label_id,
                "label_name": label_name,
                "mean_dice": mean([r["dice"] for r in label_rows]),
                "mean_hd95_mm": mean([r["hd95_mm"] for r in label_rows]),
            }
        )

    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)

    for case in sorted(by_case):
        case_rows = by_case[case]
        summary.append(
            {
                "scope": "case",
                "case": case,
                "label_id": "ALL",
                "label_name": "ALL",
                "mean_dice": mean([r["dice"] for r in case_rows]),
                "mean_hd95_mm": mean([r["hd95_mm"] for r in case_rows]),
            }
        )

    summary.append(
        {
            "scope": "overall",
            "case": "ALL",
            "label_id": "ALL",
            "label_name": "ALL",
            "mean_dice": mean([r["dice"] for r in rows]),
            "mean_hd95_mm": mean([r["hd95_mm"] for r in rows]),
        }
    )

    return summary


def print_rows(rows: list[dict]) -> None:
    print("\ncase,label_id,label_name,dice,hd95_mm,gt_voxels,pred_voxels")
    for row in rows:
        print(
            f"{row['case']},{row['label_id']},{row['label_name']},"
            f"{format_metric(row['dice'])},{format_metric(row['hd95_mm'])},"
            f"{row['gt_voxels']},{row['pred_voxels']}"
        )


def print_summary(summary: list[dict]) -> None:
    print("\nscope,case,label_id,label_name,mean_dice,mean_hd95_mm")
    for row in summary:
        print(
            f"{row['scope']},{row['case']},{row['label_id']},{row['label_name']},"
            f"{format_metric(row['mean_dice'])},{format_metric(row['mean_hd95_mm'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help='Example: "1:biatrial_wall,2:RA,3:LA". If omitted, read pred dir dataset.json.',
    )
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    labels = parse_labels(args.labels) if args.labels else load_labels(args.pred_dir)
    gt_files = list_nifti_files(args.gt_dir)

    print(f"GT_DIR={args.gt_dir}")
    print(f"PRED_DIR={args.pred_dir}")
    print(f"OUT_DIR={args.out_dir}")
    print(f"num_gt={len(gt_files)}")
    print(f"labels={labels}")

    rows = []

    for gt_path in gt_files:
        case = strip_nii_suffix(gt_path)
        pred_path = find_pred(case, args.pred_dir)

        if pred_path is None:
            print(f"[MISS] {case}: no prediction found")
            continue

        gt_img, gt = load_label_array(gt_path)
        pred_img, pred = load_label_array(pred_path)

        if gt.shape != pred.shape:
            print(f"[SHAPE_MISMATCH] {case}: gt={gt.shape}, pred={pred.shape}, pred_path={pred_path}")
            continue

        if gt.ndim != 3:
            print(f"[SKIP] {case}: expected 3D label map, got ndim={gt.ndim}")
            continue

        gt_spacing = tuple(float(x) for x in gt_img.header.get_zooms()[:3])
        pred_spacing = tuple(float(x) for x in pred_img.header.get_zooms()[:3])
        if not np.allclose(gt_spacing, pred_spacing, rtol=1e-4, atol=1e-4):
            print(f"[WARN] {case}: spacing mismatch, use gt spacing. gt={gt_spacing}, pred={pred_spacing}")

        print(f"[OK] {case}: {pred_path}")

        for label_id, label_name in labels.items():
            gt_mask = gt == label_id
            pred_mask = pred == label_id

            rows.append(
                {
                    "case": case,
                    "label_id": label_id,
                    "label_name": label_name,
                    "dice": dice_score(gt_mask, pred_mask),
                    "hd95_mm": hd95_score(gt_mask, pred_mask, gt_spacing),
                    "gt_voxels": int(gt_mask.sum()),
                    "pred_voxels": int(pred_mask.sum()),
                    "gt_path": str(gt_path),
                    "pred_path": str(pred_path),
                }
            )

    if not rows:
        print("\nNo matched GT/pred pairs. Check --gt-dir and --pred-dir.")
        return 1

    summary = build_summary(rows, labels)
    print_rows(rows)
    print_summary(summary)

    if not args.no_csv:
        per_case_csv = args.out_dir / "eval_metrics_per_case_class.csv"
        summary_csv = args.out_dir / "eval_metrics_summary.csv"

        write_csv(
            per_case_csv,
            rows,
            [
                "case",
                "label_id",
                "label_name",
                "dice",
                "hd95_mm",
                "gt_voxels",
                "pred_voxels",
                "gt_path",
                "pred_path",
            ],
        )
        write_csv(
            summary_csv,
            summary,
            ["scope", "case", "label_id", "label_name", "mean_dice", "mean_hd95_mm"],
        )

        print(f"\nSaved: {per_case_csv}")
        print(f"Saved: {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
