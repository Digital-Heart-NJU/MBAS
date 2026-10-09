#!/usr/bin/env python3
"""Build and quality-check a tetrahedral left-atrial myocardium mesh.

The input is a labelled atrial lumen and a generated outward wall.  Two nested
closed surfaces are reconstructed from signed-distance fields:

* ``endocardium``: the blood-pool/lumen boundary;
* ``epicardium``: the boundary of lumen plus the generated wall.

If ``meshtool`` is available, the region between those surfaces is filled into
an openCARP ``.pts/.elem`` mesh.  The script also writes a VTU inspection file
and a JSON report.  Mesh generation is deliberately performed only when this
script is run; importing the module has no side effects.
"""


from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
except ModuleNotFoundError:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

try:
    # Keep compatibility with the correctly-spelled module used by the
    # original pipeline.
    from step01_geometry_from_nifti import (
        DEFAULT_PARAMS,
        build_closed_wall,
        build_open_wall_candidate,
        clean_lumen,
        infer_distal_exclusion,
        load_json,
        load_lumen_mask,
        read_nifti,
        resolve_params_from_data,
        resample_to_isotropic,
    )
except ModuleNotFoundError:
    # This repository currently contains the earlier file under ``fron``.
    from step01_geometry_fron_nifti import (
        DEFAULT_PARAMS,
        build_closed_wall,
        build_open_wall_candidate,
        clean_lumen,
        infer_distal_exclusion,
        load_json,
        load_lumen_mask,
        read_nifti,
        resolve_params_from_data,
        resample_to_isotropic,
    )

from step02_surface_reconstruction import (
    extract_sdf_outer_surface,
    mesh_qc as surface_mesh_qc,
    normal_outward_fraction,
    write_surface,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


