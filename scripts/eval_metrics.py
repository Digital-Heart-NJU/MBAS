from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt

REPRO = Path("/data1/youyou/workplace/quml_repro")
GT_DIR = REPRO / "input"
PRED_DIR = REPRO / "quml_output"

labels = {1: "LA", 2: "RA", 3: "walls"}

def find_pred(case):
    candidates = [
        PRED_DIR / f"{case}_label.nii.gz",
        PRED_DIR / f"{case}.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = sorted(PRED_DIR.rglob(f"*{case}*.nii.gz"))
    return matches[0] if matches else None

def dice_iou(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    union = np.logical_or(gt, pred).sum()
    dice = 2 * inter / denom if denom else 1.0
    iou = inter / union if union else 1.0
    return dice, iou

def hd95(gt, pred, spacing):
    if not gt.any() and not pred.any():
        return 0.0
    if not gt.any() or not pred.any():
        return float("inf")
    gt_surface = np.logical_xor(gt, binary_erosion(gt))
    pred_surface = np.logical_xor(pred, binary_erosion(pred))
    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    dists = np.concatenate([dt_gt[pred_surface], dt_pred[gt_surface]])
    return float(np.percentile(dists, 95))

gt_files = sorted(GT_DIR.glob("*/*_gt.nii.gz"))
print(f"GT_DIR={GT_DIR}")
print(f"PRED_DIR={PRED_DIR}")
print(f"num_gt={len(gt_files)}")

rows = []
for gt_path in gt_files:
    case = gt_path.name.replace("_gt.nii.gz", "")
    pred_path = find_pred(case)

    if pred_path is None:
        print(f"[MISS] {case}: no prediction found")
        continue

    gt_img = nib.load(str(gt_path))
    pred_img = nib.load(str(pred_path))

    gt = np.asanyarray(gt_img.dataobj).astype(np.int16)
    pred = np.asanyarray(pred_img.dataobj).astype(np.int16)

    if gt.shape != pred.shape:
        print(f"[SHAPE_MISMATCH] {case}: gt={gt.shape}, pred={pred.shape},pred_path={pred_path}")
        continue

    spacing = gt_img.header.get_zooms()[:3]
    print(f"[OK] {case}: {pred_path}")

    for label, name in labels.items():
        d, j = dice_iou(gt == label, pred == label)
        h = hd95(gt == label, pred == label, spacing)
        rows.append((case, name, d, j, h))

print("\ncase,label,dice,iou,hd95_mm")
for case, name, d, j, h in rows:
    print(f"{case},{name},{d:.6f},{j:.6f},{h:.6f}")

if rows:
    print("\nmean_by_label")
    for name in labels.values():
        vals = [r[2] for r in rows if r[1] == name]
        print(f"{name}: mean_dice={np.mean(vals):.6f}")

    print(f"\noverall_mean_dice={np.mean([r[2] for r in rows]):.6f}")
else:
    print("\nNo matched GT/pred pairs. Check GT_DIR and PRED_DIR.")