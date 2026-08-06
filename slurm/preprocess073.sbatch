#!/bin/bash
#SBATCH --job-name=quml073_pp
#SBATCH --gres=gpu:1
#SBATCH --output=/data1/youyou/workplace/quml_repro/logs/preprocess_%j.out
#SBATCH --error=/data1/youyou/workplace/quml_repro/logs/preprocess_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/data1/youyou/conda_envs
export CONDA_PKGS_DIRS=/data1/youyou/conda_pkgs
conda activate quml
export nnUNet_raw=/data1/youyou/workplace/nnUNet_quml/nnUNet_raw
export nnUNet_preprocessed=/data1/youyou/workplace/nnUNet_quml/nnUNet_preprocessed
export nnUNet_results=/data1/youyou/workplace/nnUNet_quml/nnUNet_results

nnUNetv2_plan_and_preprocess \
    -d 73 \
    -pl nnUNetPlannerResEncM \
    -c 3d_fullres \
    -npfp 8 \
    -np 4 \
    --verify_dataset_integrity \
    --clean \
    --no_pbar
