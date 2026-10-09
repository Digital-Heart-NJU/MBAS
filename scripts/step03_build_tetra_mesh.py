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

