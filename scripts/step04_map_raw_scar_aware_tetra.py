#!/usr/bin/env python3
"""Map raw expert scar labels onto an existing tetrahedral LA wall mesh.

This is intentionally a strict mapping step. The expert scar mask is resampled
to the same isotropic geometry space with nearest-neighbour interpolation, then
tetrahedra are labelled using direct point samples in the raw scar volume.
There is no dilation, wall projection, nearest-wall snapping, or morphology
expansion. The report explicitly records any mismatch between the raw scar and
the generated myocardium wall.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
except ModuleNotFoundError:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

try:
    from step01_geometry_from_nifti import (
        build_closed_wall,
        build_open_wall_candidate,
        clean_lumen,
        infer_distal_exclusion,
        load_json,
        load_lumen_mask,
        make_isotropic_meta,
        read_nifti,
        resolve_params_from_data,
        resample_to_isotropic,
        write_nifti,
    )
except ModuleNotFoundError:
    from step01_geometry_fron_nifti import (
        build_closed_wall,
        build_open_wall_candidate,
        clean_lumen,
        infer_distal_exclusion,
        load_json,
        load_lumen_mask,
        make_isotropic_meta,
        read_nifti,
        resolve_params_from_data,
        resample_to_isotropic,
        write_nifti,
    )

from step02_surface_reconstruction import (
    count_feature_edges,
    orient_normals_outward,
    write_surface,
)
from step03_build_tetra_mesh import read_elem, read_pts, resolve_config_path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

