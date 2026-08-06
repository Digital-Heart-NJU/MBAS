from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import SimpleITK as sitk

def pick(paths):
    for p in paths:
        if p.exists():
            return p
    return None

ap = argparse.ArgumentParser()
ap.add_argument("--source",required=True)
ap.add_argument("--nnunet_raw",required=True)
ap.add_argument("--clean",action="store_true")
args = ap.parse_args()

source = Path(args.source)
dataset = Path(args.nnunet_raw) / "Dataset073_MBAS"
imagesTr = dataset / "imagesTr"
labelsTr = dataset / "labelsTr"

if args.clean and dataset.exists():
    shutil.rmtree(dataset)

imagesTr.mkdir(parents=True,exist_ok=True)
labelsTr.mkdir(parents=True,exist_ok=True)

cases = sorted([p for p in source.iterdir() if p.is_dir() and p.name.startswith("MBAS_")])
print(f"Found {len(cases)} cases")

for case in cases:
    cid = case.name
    img = pick([case / f"{cid}_image.nii.gz", case / f"{cid}_gt.nii.gz"])
    lab = pick([case / f"{cid}_label.nii.gz", case / f"{cid}_seg.nii.gz"])
    if img is None or lab is None:
        raise FileNotFoundError(f"{cid}: image={img}, label={lab}")

    shutil.copy2(img, imagesTr / f"{cid}_0000.nii.gz")

    lab_img = sitk.ReadImage(str(lab))
    arr = sitk.GetArrayFromImage(lab_img)
    arr = np.rint(arr).astype(np.uint8)

    before = sorted(np.unique(arr).tolist())
    bad = sorted(set(before) - {0, 1, 2, 3, 4})
    if bad:
        raise ValueError(f"{cid}: unexpected labels {bad}, all_labels = {before}")

    arr[arr == 4] = 3
    after = sorted(np.unique(arr).tolist())

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(lab_img)
    sitk.WriteImage(out, str(labelsTr / f"{cid}.nii.gz"))

    print(f"{cid}: {before} -> {after}")
dataset_json = {
      "name": "MBAS",
      "description": "MBAS 4-label source converted to QUM L 3-classtarget",
      "channel_names": {"0": "T2"},
      "labels": {
          "background": 0,
          "LA": 1,
          "RA": 2,
          "walls": 3
      },
      "numTraining": len(cases),
      "file_ending": ".nii.gz"
  }

with open(dataset / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Done: {dataset}")
