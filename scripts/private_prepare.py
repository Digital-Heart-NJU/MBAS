from pathlib import Path
import SimpleITK as sitk

# ===== 你自己修改这里 =====
input_dir = Path("/data1/youyou/workplace/Private_CMR")
output_dir = Path("/data1/youyou/workplace/quml_repro/private_input073")

cases = [
    {
        "case_id": "patient2",
        "image": "patient2-lgemri.nrrd",
    },
   {
        "case_id": "patient3",
        "image": "patient3-lgemri.nrrd",
    },
]
  # =========================

imagesTr = output_dir


imagesTr.mkdir(parents=True, exist_ok=True)


def convert_nrrd_to_nii_gz(src_path: Path, dst_path: Path, is_label: bool = False):
    img = sitk.ReadImage(str(src_path))

    if is_label:
        img = sitk.Cast(img, sitk.sitkUInt8)

    sitk.WriteImage(img, str(dst_path))
    print(f"Saved: {dst_path}")


for case in cases:
    case_id = case["case_id"]

    image_path = input_dir / case["image"]

    # nnU-Net 单模态图像命名：case_id_0000.nii.gz
    image_out = imagesTr / f"{case_id}_0000.nii.gz"
    convert_nrrd_to_nii_gz(image_path, image_out, is_label=False)

    
