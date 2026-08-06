#!/bin/bash
#SBATCH --job-name=quml_eval
#SBATCH --output=/data1/youyou/workplace/quml_repro/logs/slurm_eval_all_%j.out
#SBATCH --error=/data1/youyou/workplace/quml_repro/logs/slurm_eval_all_%j.err

set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate quml

REPRO=/data1/youyou/workplace/quml_repro
DJFILE=/data1/youyou/workplace/nnUNet_quml/nnUNet_results/Dataset073_MBAS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/dataset.json
PFILE=/data1/youyou/workplace/nnUNet_quml/nnUNet_results/Dataset073_MBAS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/plans.json
export NNUNET_SRC=/data1/youyou/workplace/nnUNet-master
export PLAN_DIR=$REPRO/nnUNet_results/Dataset073_MBAS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres

export PYTHONPATH=$NNUNET_SRC:$PYTHONPATH
export nnUNet_results=$REPRO/nnUNet_results
export nnUNet_raw=$REPRO/nnUNet_raw
export nnUNet_preprocessed=$REPRO/nnUNet_preprocessed

mkdir -p $REPRO/logs
rm -rf $REPRO/eval_gt_nnunet $REPRO/eval_pred_nnunet
mkdir -p $REPRO/eval_gt_nnunet $REPRO/eval_pred_nnunet

for gt in $REPRO/quml_input/*/*_gt.nii.gz; do
  case=$(basename "$gt" _gt.nii.gz)

  pred=""
  for cand in \
    "$REPRO/quml_output/${case}_label.nii.gz" \
    "$REPRO/quml_output/${case}.nii.gz" \
    "$REPRO/quml_output/${case}_gt_label.nii.gz"
  do
    if [ -f "$cand" ]; then
      pred="$cand"
      break
    fi
  done

  if [ -n "$pred" ]; then
    ln -s "$(readlink -f "$gt")" "$REPRO/eval_gt_nnunet/${case}.nii.gz"
    ln -s "$(readlink -f "$pred")" "$REPRO/eval_pred_nnunet/${case}.nii.gz"
    echo "[OK] $case -> $pred"
  else
    echo "[MISS] $case"
  fi
done

  echo "GT count:"
  find $REPRO/eval_gt_nnunet -name "*.nii.gz" | wc -l

  echo "PRED count:"
  find $REPRO/eval_pred_nnunet -name "*.nii.gz" | wc -l

 nnUNetv2_evaluate_folder \
    "${REPRO}/eval_gt_nnunet" \
    "${REPRO}/eval_pred_nnunet" \
    -djfile "${DJFILE}" \
    -pfile "${PFILE}" \
    -o "${REPRO}/logs/eval_all_summary.json" \
    2>&1 | tee "${REPRO}/logs/eval_all.log"


  python3 - <<'PY'
  import json
  p="/data1/youyou/workplace/quml_repro/logs/eval_all_summary.json"
  d=json.load(open(p))

  print("\nforeground_mean:")
  print(d.get("foreground_mean"))

  print("\nmean:")
  for k, v in d.get("mean", {}).items():
      print(k, v)
  PY
