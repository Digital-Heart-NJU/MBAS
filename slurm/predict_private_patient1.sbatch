#!/bin/bash
#SBATCH --job-name=quml_p1_pred
#SBATCH --gres=gpu:1
#SBATCH --output=/data1/youyou/workplace/quml_repro/logs/private_p1_predict_%j.out
#SBATCH --error=/data1/youyou/workplace/quml_repro/logs/private_p1_predict_%j.err
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh

export nnUNet_raw=/data1/youyou/workplace/nnUNet_quml/nnUNet_raw
export nnUNet_preprocessed=/data1/youyou/workplace/nnUNet_quml/nnUNet_preprocessed
export nnUNet_results=/data1/youyou/workplace/nnUNet_quml/nnUNet_results

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
rm -rf /data1/youyou/workplace/quml_repro/private_pred073
mkdir -p /data1/youyou/workplace/quml_repro/private_pred073


nnUNetv2_predict \
  -i /data1/youyou/workplace/quml_repro/private_input073 \
  -o /data1/youyou/workplace/quml_repro/private_pred073 \
  -d 73 \
  -c 3d_fullres \
  -tr nnUNetTrainer \
  -p nnUNetResEncUNetMPlans \
  -f 0 1 2 3 4 \
  -chk checkpoint_final.pth \
  -npp 4 \
  -nps 4 \
  --disable_progress_bar

