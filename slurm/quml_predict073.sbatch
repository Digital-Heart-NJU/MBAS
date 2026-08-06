#!/bin/bash
#SBATCH --job-name=quml_pred
#SBATCH --gres=gpu:2
#SBATCH --output=/data1/youyou/workplace/quml_repro/logs/quml_predict_%j.out
#SBATCH --error=/data1/youyou/workplace/quml_repro/logs/quml_predict_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh

export nnUNet_raw=/data1/youyou/workplace/nnUNet_quml/nnUNet_raw
export nnUNet_preprocessed=/data1/youyou/workplace/nnUNet_quml/nnUNet_preprocessed
export nnUNet_results=/data1/youyou/workplace/nnUNet_quml/nnUNet_results

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /data1/youyou/workplace/1-QUML
python predict.py \
    --input_dir /data1/youyou/workplace/quml_repro/gtTs073_mapped/ \
    --output_dir /data1/youyou/workplace/quml_repro/quml_output/ \
    --model_path /data1/youyou/workplace/nnUNet_quml/nnUNet_results/
