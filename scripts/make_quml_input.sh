#!/bin/bash
  set -e

  PROJECT=/data1/youyou/workplace/quml_repro
  TEST_SRC=/data1/youyou/workplace/MBAS_Testing_4C

  rm -rf "$PROJECT/quml_input" "$PROJECT/quml_output"
  mkdir -p "$PROJECT/quml_input" "$PROJECT/quml_output"

  for case in "$TEST_SRC"/MBAS_*; do
    cid=$(basename "$case")
    img="$case/${cid}_image.nii.gz"

    if [ ! -f "$img" ]; then
      img="$case/${cid}_gt.nii.gz"
    fi

    if [ -f "$img" ]; then
      mkdir -p "$PROJECT/quml_input/$cid"
      cp "$img" "$PROJECT/quml_input/$cid/${cid}_gt.nii.gz"
      echo "$cid"
    else
      echo "missing image: $cid"
    fi
  done
