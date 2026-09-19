from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from scipy import ndimage as ndi
try:
    import vtkmodules.all as vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
except ModuleNotFoundError:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

CONNECTTIVITY_26 = np.ones((3,3,3),dtype=bool)

DEFAULT_PARAMS = {
    "atrium_label": "auto",
    "atrium_use_nonzero_if_label_missing": True,
    "fill_holes": True,
    "close_radius_mm": 0.0,
    "geometry_workspace_spacing_mm": "auto",
    "wall_out_mm": None,
}

@dataclass
class NiftiMeta:
    dims_xyz:tuple[int,int,int]
    spacing_xyz:tuple[float,float,float]
    origin_xyz:tuple[float,float,float]
    qform:vtk.vtkMatrix4x4 | None
    sform:vtk.vtkMatrix4x4 | None

    @property
    def spacing_zyx(self) -> tuple[float,float,float]:
        sx, sy, sz = self.spacing_xyz
        return (sz, sy, sx)

    @property
    def voxel_volume_mm3(self) -> float:
        sx, sy, sz = self.spacing_xyz
        return sx * sy * sz


def load_json(path:Path) -> dict[str,Any]:
    with path.open("r",encoding="utf-8") as f:
        return json.load(f)


def write_json(path:Path,payload:dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        json.dump(payload,f,indent=2)
        f.write("\n")


def parse_atrium_label(value:Any) -> Any:
    if value is None:
        return "auto"
    if isinstance(value,(int,float)):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"auto","infer"}:
        return "auto"
    if lowered in {"none","nonzero","all_nonzero"}:
        return None
    try:
        return int(value)
    except ValueError:
        return float(value)


def infer_atrium_label(atrium:np.ndarray) -> int | float:
    values,counts = np.unique(atrium,return_counts=True)
    nonzero = [(value,count) for value,count in zip(values,counts) if value != 0]
    if not nonzero:
        raise ValueError("Cannot infer atrium label: segmentation has no nonzero voxels")
    label,_ = max(nonzero,key=lambda item:item[1])
    label = label.item() if isinstance(label,np.generic) else label
    if isinstance(label,float) and label.is_integer():
        return int(label)
    return label


def resolve_params_from_data(
    params:dict[str,Any],
    atrium:np.ndarray,
    source_meta:NiftiMeta,
) -> dict[str,Any]:
    params = dict(params)

    atrium_label = parse_atrium_label(params.get("atrium_label","auto"))
    if atrium_label == "auto":
        atrium_label = infer_atrium_label(atrium)
    params["atrium_label"] = atrium_label

    target_spacing = params.get("geometry_workspace_spacing_mm","auto")
    if isinstance(target_spacing,str) and target_spacing.lower() in {"auto","infer","min_input_spacing"}:
        target_spacing = min(source_meta.spacing_zyx)
    params["geometry_workspace_spacing_mm"] = float(target_spacing)

    if params.get("wall_out_mm") is None:
        raise ValueError(
            "wall_out_mm cannot be inferred from data. "
            "Please pass --wall-out-mm explicitly."
        )
    params["wall_out_mm"] = float(params["wall_out_mm"])
    params["close_radius_mm"] = float(params.get("close_radius_mm",0.0))
    params["fill_holes"] = bool(params.get("fill_holes",True))
    params["atrium_use_nonzero_if_label_missing"] = bool(
        params.get("atrium_use_nonzero_if_label_missing",True)
    )
    params.pop("auto_opening",None)
    return params

def ellipsoid_structure(radius_mm:float,spacing_zyx:tuple[float,float,float]) -> np.ndarray:
    if radius_mm <= 0:
        return np.ones((1,1,1),dtype=bool) #半径≤0，返回单个体素的True结构元（相当于不做膨胀）。
     #计算各轴需要的半宽（体素个数）
    rz, ry, rx = [max(1, int(math.ceil(radius_mm / s))) for s in spacing_zyx]
    zz, yy, xx = np.ogrid[-rz: rz + 1, -ry: ry + 1, -rx: rx + 1]
    dist2 = (
        (zz * spacing_zyx[0]) ** 2
        + (yy * spacing_zyx[1]) ** 2
        + (xx * spacing_zyx[2]) ** 2
    )
    return dist2 <= radius_mm**2


def read_nifti(path:Path) -> tuple[np.ndarray,NiftiMeta]:
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(path))
    reader.Update()

    image = reader.GetOutput()
    dims_xyz = tuple(int(v) for v in image.GetDimensions())
    spacing_xyz = tuple(float(v) for v in image.GetSpacing())
    origin_xyz = tuple(float(v) for v in image.GetOrigin())
    scalars = image.GetPointData().GetScalars()
    if scalars is None:
        raise ValueError(f"No scalar data found in {path}")

    arr = vtk_to_numpy(scalars).reshape(
        (dims_xyz[2],dims_xyz[1],dims_xyz[0])
    ).copy()
    return arr,NiftiMeta(
        dims_xyz=dims_xyz,
        spacing_xyz=spacing_xyz,
        origin_xyz=origin_xyz,
        qform=reader.GetQFormMatrix(),
        sform=reader.GetSFormMatrix(),
    )


def load_lumen_mask(atrium:np.ndarray,params:dict[str,Any]) -> tuple[np.ndarray,str]:
    atrium_label = params.get("atrium_label")
    if atrium_label is None:
        lumen = atrium != 0
        source = "nonzero"
    else:
        lumen = atrium == atrium_label
        source = f"label_{atrium_label}"
        if(
            not lumen.any()
            and params.get("atrium_use_nonzero_if_label_missing",True)
            and np.count_nonzero(atrium) > 0
        ):
            lumen = atrium !=0
            source = f"nonzero_fallback_label_{atrium_label}_missing"
    if not lumen.any():
        raise ValueError("Empty atrium mask after label selection")
    return lumen,source


def component_stats(mask:np.ndarray,voxel_volume_mm3:float) -> tuple[np.ndarray,list[dict[str,Any]]]:
    labels,count = ndi.label(mask,structure=CONNECTTIVITY_26)
    if count == 0:
        return labels, []
    sizes = np.bincount(labels.ravel())
    stats: list[dict[str, Any]] = []
    for label_id in range(1, count + 1):
        voxels = int(sizes[label_id])
        if voxels == 0:
            continue
        stats.append({
            "label_id":int(label_id),
            "voxels":voxels,
            "ml":float(voxels * voxel_volume_mm3 / 1000.0),
        })
    stats.sort(key=lambda item: item["voxels"],reverse=True)
    return labels,stats


def keep_largest_component(mask:np.ndarray, voxel_volume_mm3:float) -> np.ndarray:
    labels, stats = component_stats(mask,voxel_volume_mm3)
    if not stats:
        return mask & False
    return labels == stats[0]["label_id"]


def clean_lumen(lumen:np.ndarray,
                spacing_zyx:tuple[float,float,float],
                voxel_volume_mm3:float,
                params:dict[str,Any]) -> np.ndarray:
    lumen = keep_largest_component(lumen,voxel_volume_mm3)
    if params.get("fill_holes", True):
        lumen = ndi.binary_fill_holes(lumen)
    close_radius = float(params.get("close_radius_mm", 0.0))
    if close_radius > 0:
        lumen = ndi.binary_closing(
            lumen,
            structure=ellipsoid_structure(close_radius, spacing_zyx),
        )
        lumen = keep_largest_component(lumen, voxel_volume_mm3)
    return lumen

#把 3D mask 重采样成各向同性体素（三轴 spacing 相同）
def resample_to_isotropic(
        mask: np.ndarray,
        source_spacing_zyx: tuple[float, float, float],
        target_spacing_mm: float,
) -> tuple[np.ndarray, tuple[float, float, float], list[float]]:
    target_spacing_zyx = (target_spacing_mm, target_spacing_mm, target_spacing_mm)
    zoom = [source_spacing_zyx[i] / target_spacing_zyx[i] for i in range(3)]
    if all(abs(v - 1.0) < 1e-8 for v in zoom):
        return mask.astype(bool, copy=True), target_spacing_zyx, zoom
    ## order=0：最近邻插值；先转uint8缩放，再>0转回bool mask
    return ndi.zoom(mask.astype(np.uint8), zoom=zoom, order=0) > 0, target_spacing_zyx, zoom


def build_closed_wall(lumen: np.ndarray, spacing_zyx: tuple[float, float, float], wall_out_mm: float)-> np.ndarray:
    ## distance_transform_edt：计算每个背景体素【到最近的lumen边界】的欧氏距离（单位mm）
    outside_dist = ndi.distance_transform_edt(~lumen, sampling=spacing_zyx)
    return (~lumen) & (outside_dist > 0) & (outside_dist <= wall_out_mm)


def infer_distal_exclusion(
    lumen: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    voxel_volume_mm3: float,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    auto = params.get("auto_opening", {})
    if not auto.get("enabled", False):
        return lumen & False, {"enabled": False}

    core_radius = float(auto.get("body_core_erode_mm", 5.0))
    discard_distance = float(auto.get("vessel_discard_distance_from_core_mm", 8.0))
    min_discard_ml = float(auto.get("min_discard_component_ml", 0.1))
    preserve_laa = bool(auto.get("preserve_largest_distal_component_as_laa", True))

    # # Step1：腐蚀心房腔，得到心房主体核心core
    core = ndi.binary_erosion(lumen, structure=ellipsoid_structure(core_radius, spacing_zyx))
    core = keep_largest_component(core, voxel_volume_mm3)
    if not core.any():
        inside_dist = ndi.distance_transform_edt(lumen, sampling=spacing_zyx)
        core = keep_largest_component(lumen & (inside_dist >= core_radius), voxel_volume_mm3)
    if not core.any():
        return lumen & False, {"enabled": True, "error": "empty_body_core"}

    # Step2：计算lumen内每个体素到心房core的最短欧氏距离（mm）
    dist_to_core = ndi.distance_transform_edt(~core, sampling=spacing_zyx)
    distal = lumen & (dist_to_core >= discard_distance)
    # Step3：远端区域连通域分析
    labels, stats = component_stats(distal, voxel_volume_mm3)

    preserved_id: int | None = None
    if preserve_laa and len(stats) >= 2:
        preserved_id = int(stats[0]["label_id"])

    # Step4：筛选需要剔除的连通域
    selected_ids: list[int] = []
    for item in stats:
        label_id = int(item["label_id"])
        if label_id == preserved_id:
            continue
        if float(item["ml"]) >= min_discard_ml:
            selected_ids.append(label_id)

    exclusion = np.isin(labels, selected_ids)
    info = {
        "enabled": True,
        "distal_component_count": len(stats),
        "preserved_laa_candidate_label_id": preserved_id,
        "selected_exclusion_label_ids": selected_ids,
        "exclusion_voxels": int(exclusion.sum()),
        "exclusion_ml": float(exclusion.sum() * voxel_volume_mm3 / 1000.0),
    }
    return exclusion, info


def build_open_wall_candidate(
    closed_wall: np.ndarray,
    exclusion_lumen: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    params: dict[str, Any],
) -> np.ndarray:
    auto = params.get("auto_opening", {})
    margin = float(auto.get("opening_block_margin_mm", 0.8))
    radius = float(params["wall_out_mm"]) + margin
    block = ndi.binary_dilation(exclusion_lumen, structure=ellipsoid_structure(radius, spacing_zyx))
    return closed_wall & ~block


def clone_scaled_matrix(matrix: vtk.vtkMatrix4x4 | None, scale_xyz: tuple[float, float, float]) -> vtk.vtkMatrix4x4 | None:
    if matrix is None:
        return None
    out = vtk.vtkMatrix4x4()
    out.DeepCopy(matrix)
    for col, scale in enumerate(scale_xyz):
        for row in range(3):# row 0,1,2（只处理前3行，跳过第4行）
            out.SetElement(row, col, out.GetElement(row, col) * scale)
    return out


def make_isotropic_meta(
        source_meta: NiftiMeta,
        arr_zyx: np.ndarray,
        spacing_mm: float,
) -> NiftiMeta:
    nz, ny, nx = arr_zyx.shape
    old_sx, old_sy, old_sz = source_meta.spacing_xyz
    new_spacing_xyz = (spacing_mm, spacing_mm, spacing_mm)
    scale_xyz = (
        new_spacing_xyz[0] / old_sx,
        new_spacing_xyz[1] / old_sy,
        new_spacing_xyz[2] / old_sz,
    )
    return NiftiMeta(
        dims_xyz=(nx, ny, nz),
        spacing_xyz=new_spacing_xyz,
        origin_xyz=source_meta.origin_xyz,
        qform=clone_scaled_matrix(source_meta.qform, scale_xyz),
        sform=clone_scaled_matrix(source_meta.sform, scale_xyz),
    )


def write_nifti(path: Path, arr_zyx: np.ndarray, meta: NiftiMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    arr_u8 = np.ascontiguousarray(arr_zyx.astype(np.uint8, copy=False))
    nz, ny, nx = arr_u8.shape

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(meta.spacing_xyz)
    image.SetOrigin(meta.origin_xyz)

    vtk_arr = numpy_to_vtk(
        arr_u8.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    vtk_arr.SetName("wall_mask")
    image.GetPointData().SetScalars(vtk_arr)

    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(image)
    writer.SetNIFTIVersion(1)
    if meta.qform is not None:
        writer.SetQFormMatrix(meta.qform)
    if meta.sform is not None:
        writer.SetSFormMatrix(meta.sform)
    writer.Write()


def mesh_boundary_qc(mask: np.ndarray, spacing_zyx: tuple[float, float, float]) -> dict[str, int | None]:
    if not mask.any():
        return {"surface_cells": 0, "boundary_edges": None, "non_manifold_edges": None}

    arr = np.ascontiguousarray(mask.astype(np.uint8))
    nz, ny, nx = arr.shape
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    sz, sy, sx = spacing_zyx
    image.SetSpacing(sx, sy, sz)
    vtk_arr = numpy_to_vtk(arr.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    image.GetPointData().SetScalars(vtk_arr)

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(image)
    contour.SetValue(0, 1)
    contour.Update()
    surface = contour.GetOutput()

    boundary = vtk.vtkFeatureEdges()
    boundary.SetInputData(surface)
    boundary.BoundaryEdgesOn()
    boundary.NonManifoldEdgesOff()
    boundary.FeatureEdgesOff()
    boundary.ManifoldEdgesOff()
    boundary.Update()

    non_manifold = vtk.vtkFeatureEdges()
    non_manifold.SetInputData(surface)
    non_manifold.BoundaryEdgesOff()
    non_manifold.NonManifoldEdgesOn()
    non_manifold.FeatureEdgesOff()
    non_manifold.ManifoldEdgesOff()
    non_manifold.Update()

    return {
        "surface_cells": int(surface.GetNumberOfCells()),
        "boundary_edges": int(boundary.GetOutput().GetNumberOfCells()),
        "non_manifold_edges": int(non_manifold.GetOutput().GetNumberOfCells()),
    }


def build_case(case:str,root:Path,output_root:Path,params:dict[str,Any]) -> dict[str,Any]:
    case_dir = root / case
    atrium_path = case_dir / "atriumSegImgMO.nii.gz"
    if not atrium_path.exists():
        raise FileNotFoundError(atrium_path)

    atrium,source_meta = read_nifti(atrium_path)
    params = resolve_params_from_data(params,atrium,source_meta)
    source_spacing = source_meta.spacing_zyx
    source_voxel_volume = source_meta.voxel_volume_mm3

    lumen,mask_source = load_lumen_mask(atrium,params)
    lumen = clean_lumen(lumen,source_spacing,source_voxel_volume,params)

    target_spacing_mm = float(params["geometry_workspace_spacing_mm"])
    lumen_iso, iso_spacing, zoom_factors = resample_to_isotropic(lumen, source_spacing, target_spacing_mm)
    iso_voxel_volume = float(np.prod(iso_spacing))
    lumen_iso = clean_lumen(lumen_iso, iso_spacing, iso_voxel_volume, params)

    closed_wall = build_closed_wall(lumen_iso, iso_spacing, float(params["wall_out_mm"]))

    iso_meta = make_isotropic_meta(source_meta, closed_wall, target_spacing_mm)
    case_out = output_root / case
    closed_file = case_out / f"{case}_closed_wall_mask.nii.gz"
    params_file = case_out / f"{case}_effective_params.json"
    summary_file = case_out / f"{case}_step01_summary.json"
    write_nifti(closed_file, closed_wall, iso_meta)

    closed_labels, closed_components = component_stats(closed_wall, iso_voxel_volume)
    del closed_labels

    summary = {
        "case": case,
        "mask_source": mask_source,
        "output_files": {
            "closed_wall_mask": str(closed_file),
            "effective_params": str(params_file),
            "summary": str(summary_file),
        },
        "effective_params": params,
        "source_shape_zyx": [int(v) for v in atrium.shape],
        "output_shape_zyx": [int(v) for v in closed_wall.shape],
        "source_spacing_zyx_mm": [float(v) for v in source_spacing],
        "output_spacing_zyx_mm": [float(v) for v in iso_spacing],
        "zoom_factors_zyx": [float(v) for v in zoom_factors],
        "closed_wall": {
            "voxels": int(closed_wall.sum()),
            "ml": float(closed_wall.sum() * iso_voxel_volume / 1000.0),
            "component_count": len(closed_components),
            "mesh_qc": mesh_boundary_qc(closed_wall, iso_spacing),
        },
    }
    write_json(params_file,params)
    write_json(summary_file,summary)
    return summary
    
    


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",default="train_1")
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output-root",type=Path,default=Path("step01_outputs/two_wall_masks"))
    parser.add_argument("--config",type=Path,default=None,help="Optional JSON parameter file")
    parser.add_argument("--wall-out-mm",type=float,required=True,help="Fixed outward wall ROI thickness in mm")
    parser.add_argument("--close-radius-mm",type=float,default=None,help="Morphological closing radius in mm")
    parser.add_argument("--target-spacing-mm",type=float,default=None,help="Output isotropic spacing in mm")
    parser.add_argument("--atrium-label",default=None,help="Use auto, nonzero, or a numeric segmentation label")
    parser.add_argument("--fill-holes",dest="fill_holes",action="store_true",default=None)
    parser.add_argument("--no-fill-holes",dest="fill_holes",action="store_false")
    args = parser.parse_args()

    params = dict(DEFAULT_PARAMS)
    if args.config is not None:
        params.update(load_json(args.config))

    params["wall_out_mm"] = args.wall_out_mm
    if args.close_radius_mm is not None:
        params["close_radius_mm"] = args.close_radius_mm
    if args.target_spacing_mm is not None:
        params["geometry_workspace_spacing_mm"] = args.target_spacing_mm
    if args.atrium_label is not None:
        params["atrium_label"] = parse_atrium_label(args.atrium_label)
    if args.fill_holes is not None:
        params["fill_holes"] = args.fill_holes

    summary = build_case(
        case = args.case,
        root = args.root.resolve(),
        output_root = args.output_root.resolve(),
        params = params,
    )
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
