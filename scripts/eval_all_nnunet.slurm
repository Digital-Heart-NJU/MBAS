#!/bin/bash
#SBATCH --job-name=quml_eval073
#SBATCH --output=/data1/youyou/workplace/quml_repro/logs/slurm_eval_metrics_%j.out
#SBATCH --error=/data1/youyou/workplace/quml_repro/logs/slurm_eval_metrics_%j.err

set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate quml

REPRO=/data1/youyou/workplace/quml_repro
GT_DIR=$REPRO/gtTs073_mapped
PRED_DIR=$REPRO/predTs073
OUT_DIR=$REPRO/logs

mkdir -p "$OUT_DIR"

python3 "$REPRO/scripts/eval_metrics.py"   --gt-dir "$GT_DIR"   --pred-dir "$PRED_DIR"   --out-dir "$OUT_DIR"   2>&1 | tee "$OUT_DIR/eval_metrics073.log"
