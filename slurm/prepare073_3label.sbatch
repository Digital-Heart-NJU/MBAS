#!/bin/bash
#SBATCH --job-name=quml073_prep
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --output=/data1/youyou/workplace/quml_repro/logs/prepare_%j.out

#SBATCH --error=/data1/youyou/workplace/quml_repro/logs/prepare_%j.err
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/data1/youyou/conda_envs
export CONDA_PKGS_DIRS=/data1/youyou/conda_pkgs
python /data1/youyou/workplace/quml_repro/scripts/prepare_dataset073_mbas_3label.py \
  --train-source /data1/youyou/workplace/MBAS_Training_4C \
  --test-source /data1/youyou/workplace/MBAS_Testing_4C \
  --nnunet-raw /data1/youyou/workplace/nnUNet_quml/nnUNet_raw \
  --repro /data1/youyou/workplace/quml_repro \
  --clean




