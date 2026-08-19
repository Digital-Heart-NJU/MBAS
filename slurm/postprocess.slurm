#!/bin/bash
#SBATCH --job-name=pp_private35
#SBATCH --cpus-per-task=4
#SBATCH --output=/data1/youyou/workplace/quml_repro/logs/postprocess_private_close35_%j.out
#SBATCH --error=/data1/youyou/workplace/quml_repro/logs/postprocess_private_close35_%j.err

set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate quml

REPRO=/data1/youyou/workplace/quml_repro

mkdir -p "$REPRO/logs"

python3 "$REPRO/scripts/postprocess_private_pred073.py" \
    --output-dir "$REPRO/private_pred073_postprocessed_close7" \
    --close-mm 7 \
    --max-wall-patch-voxels 100000 \
    --max-internal-hole-voxels 100000 \
    --overwrite
