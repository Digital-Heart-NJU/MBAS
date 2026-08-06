#!/bin/bash
#SBATCH --job-name=quml073_train
#SBATCH --array=0-4

#SBATCH --gres=gpu:1
#SBATCH --output=/data1/youyou/workplace/quml_repro/logs/train_f%a_%j.out

#SBATCH --error=/data1/youyou/workplace/quml_repro/logs/train_f%a_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/data1/youyou/conda_envs
export CONDA_PKGS_DIRS=/data1/youyou/conda_pkgs
conda activate nnunetv2
export nnUNet_raw=/data1/youyou/workplace/nnUNet_quml/nnUNet_raw
export nnUNet_preprocessed=/data1/youyou/workplace/nnUNet_quml/nnUNet_preprocessed
export nnUNet_results=/data1/youyou/workplace/nnUNet_quml/nnUNet_results
nnUNetv2_train 73 3d_fullres ${SLURM_ARRAY_TASK_ID} \
    -tr nnUNetTrainer \
    -p nnUNetResEncUNetMPlans \
    --npz
