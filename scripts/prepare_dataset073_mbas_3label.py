from pathlib import Path
import argparse, json, shutil
import numpy as np
import SimpleITK as sitk

def pick(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def map_label(label_path, out_path):
    img = sitk.ReadImage(str(label_path))
    arr = np.rint(sitk.GetArrayFromImage(img)).astype(np.int16)
    before = sorted(np.unique(arr).tolist())
    bad = sorted(set(before) - {0, 1, 2, 3, 4})
    if bad:
        raise ValueError(f"{label_path}: unexpected labels {bad}; all—labels={before}")

    mapped = np.zeros(arr.shape, dtype=np.uint8)
    mapped[(arr == 1) | (arr == 2)] = 1
    mapped[arr == 3] = 2
    mapped[arr == 4] = 3


    out = sitk.GetImageFromArray(mapped)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(out_path))
    return before, sorted(np.unique(mapped).tolist())

ap = argparse.ArgumentParser()
ap.add_argument("--train-source", required=True)
ap.add_argument("--test-source", required=True)
ap.add_argument("--nnunet-raw", required=True)
ap.add_argument("--repro", required=True)
ap.add_argument("--clean", action="store_true")
args = ap.parse_args()

train_src = Path(args.train_source)
test_src = Path(args.test_source)
nnunet_raw = Path(args.nnunet_raw)
repro = Path(args.repro)

dataset = nnunet_raw / "Dataset073_MBAS"
imagesTr = dataset / "imagesTr"
labelsTr = dataset / "labelsTr"
imagesTs = repro / "imagesTs073"
gtTs = repro / "gtTs073_mapped"

if args.clean:
    for p in [dataset, imagesTs, gtTs, repro / "predTs073"]:
        if p.exists():
            shutil.rmtree(p)

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)
imagesTs.mkdir(parents=True, exist_ok=True)
gtTs.mkdir(parents=True, exist_ok=True)

train_cases = sorted(p for p in train_src.iterdir() if p.is_dir() and
p.name.startswith("MBAS_"))
test_cases = sorted(p for p in test_src.iterdir() if p.is_dir() and
p.name.startswith("MBAS_"))

for case in train_cases:
    cid = case.name
    img = pick([case / f"{cid}_image.nii.gz", case / f"{cid}_gt.nii.gz"])
    lab = pick([case / f"{cid}_label.nii.gz", case / f"{cid}_seg.nii.gz"])
    if img is None or lab is None:
        raise FileNotFoundError(cid)
    shutil.copy2(img, imagesTr / f"{cid}_0000.nii.gz")
    before, after = map_label(lab, labelsTr / f"{cid}.nii.gz")
    print(f"TR {cid}: {before} -> {after}")

for case in test_cases:
    cid = case.name
    img = pick([case / f"{cid}_image.nii.gz", case / f"{cid}_gt.nii.gz"])
    lab = pick([case / f"{cid}_label.nii.gz", case / f"{cid}_seg.nii.gz"])
    if img is None:
        raise FileNotFoundError(cid)
    shutil.copy2(img, imagesTs / f"{cid}_0000.nii.gz")
    if lab is not None:
        before, after = map_label(lab, gtTs / f"{cid}.nii.gz")
        print(f"TS {cid}: {before} -> {after}")

dataset_json = {
    "name": "MBAS",
    "channel_names": {"0": "T2"},
    "labels": {
        "background": 0,
        "biatrail_wall": 1,
        "RA": 2,
        "LA": 3
     },
    "numTraining": len(train_cases),
    "numTest": len(test_cases),
    "file_ending": ".nii.gz"
}
with open(dataset / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Done: {dataset}")
print(f"Test images: {imagesTs}")
print(f"Mapped test labels: {gtTs}")

