import re
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import subprocess
import torch
from typing import Tuple

def _S_m11_to_01(device, dtype):
    # Map [-1,1] -> [0,1] in homogeneous form
    # x01 = 0.5 * x_m11 + 0.5
    S_inv = torch.tensor([[0.5, 0. , 0. , 0.5],
                          [0. , 0.5, 0. , 0.5],
                          [0. , 0. , 0.5, 0.5],
                          [0. , 0. , 0. , 1. ]], device=device, dtype=dtype)
    # Map [0,1] -> [-1,1]: x_m11 = 2*x01 - 1
    S = torch.tensor([[2., 0., 0., -1.],
                      [0., 2., 0., -1.],
                      [0., 0., 2., -1.],
                      [0., 0., 0.,  1.]], device=device, dtype=dtype)
    return S, S_inv

def _S_vox_to_01(size_xyz: Tuple[int,int,int], device, dtype):
    # voxel index (0..W-1, 0..H-1, 0..D-1)  -> [0,1]
    X, Y, Z = size_xyz
    sx = 1.0 / (X - 1) if X > 1 else 1.0
    sy = 1.0 / (Y - 1) if Y > 1 else 1.0
    sz = 1.0 / (Z - 1) if Z > 1 else 1.0
    S_vox2_01  = torch.tensor([[sx, 0., 0., 0.],
                               [0., sy, 0., 0.],
                               [0., 0., sz, 0.],
                               [0., 0.,  0., 1.]], device=device, dtype=dtype)
    S_01_to_vox = torch.tensor([[1./sx if X>1 else 1., 0., 0., 0.],
                                [0., 1./sy if Y>1 else 1., 0., 0.],
                                [0., 0., 1./sz if Z>1 else 1., 0.],
                                [0., 0., 0., 1.]], device=device, dtype=dtype)
    return S_vox2_01, S_01_to_vox

def adjust_theta_to_unit_cube(
    theta44: torch.Tensor,                      # [N,4,4], mapping fixed -> moving in the given 'space'
    space: str = "minus1_1",                    # '01' | 'minus1_1' | 'vox'
    size_fix_xyz: Tuple[int,int,int] = None,    # needed if space == 'vox'
    size_mov_xyz: Tuple[int,int,int] = None     # needed if space == 'vox'
) -> torch.Tensor:
    """
    Convert a batch of 4x4 affines so they act on coordinates in [0,1]^3.

    Inputs
    ------
    theta44 : [N,4,4]
        Affine mapping from fixed to moving in the coordinate 'space' you specify.
    space : {'01','minus1_1','vox'}
        - '01'        : theta is already defined on [0,1] coords → returned unchanged.
        - 'minus1_1'  : theta maps fixed[-1,1] -> moving[-1,1]. We return an
                        equivalent theta that maps fixed[0,1] -> moving[0,1].
        - 'vox'       : theta maps fixed_voxel -> moving_voxel. You must provide
                        size_fix_xyz=(Xf,Yf,Zf), size_mov_xyz=(Xm,Ym,Zm).
                        We return an equivalent theta that maps fixed[0,1] -> moving[0,1].
    size_fix_xyz, size_mov_xyz :
        Image sizes in XYZ order. Required when space == 'vox'.

    Returns
    -------
    theta01_44 : [N,4,4]
        Affine mapping fixed_[0,1] -> moving_[0,1], ready to apply to a meshgrid
        built directly in [0,1] (with homogeneous coords [x,y,z,1]^T).
    """
    if theta44.ndim != 3 or theta44.shape[1:] != (4,4):
        raise ValueError(f"theta44 must be [N,4,4], got {tuple(theta44.shape)}")

    device = theta44.device
    dtype  = theta44.dtype
    N      = theta44.shape[0]

    if space == "01":
        return theta44  # already good to go

    if space == "minus1_1":
        # theta_m11 acts on [-1,1]; convert to act on [0,1]: theta_01 = S_inv * theta_m11 * S
        S, S_inv = _S_m11_to_01(device, dtype)
        # batch @: (4x4)@(N,4,4)@(4x4) via expand + bmm
        left  = S_inv.expand(N, 4, 4)
        right = S.expand(N, 4, 4)
        return left.bmm(theta44).bmm(right)

    if space == "vox":
        if size_fix_xyz is None or size_mov_xyz is None:
            raise ValueError("size_fix_xyz and size_mov_xyz must be provided when space='vox'.")

        S_fix_vox2_01, S_fix_01_to_vox = _S_vox_to_01(size_fix_xyz, device, dtype)
        S_mov_vox2_01, S_mov_01_to_vox = _S_vox_to_01(size_mov_xyz, device, dtype)

        # We have: x_mov_vox = theta_vox @ x_fix_vox
        # Want:    x_mov_01  = theta_01  @ x_fix_01
        # With x_fix_vox = S_fix_01_to_vox @ x_fix_01  and  x_mov_01 = S_mov_vox2_01 @ x_mov_vox
        # => theta_01 = S_mov_vox2_01 @ theta_vox @ S_fix_01_to_vox
        left  = S_mov_vox2_01.expand(N, 4, 4)
        right = S_fix_01_to_vox.expand(N, 4, 4)
        return left.bmm(theta44).bmm(right)

    raise ValueError("space must be one of {'01','minus1_1','vox'}")
# ---- 1) Read the 4x4 RAS->RAS block from a .lta ----
def read_lta_ras_to_ras(lta_path: str) -> np.ndarray:
    with open(lta_path, "r") as f:
        txt = f.read()
    m = re.search(r"1\s+4\s+4\s+([\s\S]*?)\nsrc volume info", txt)
    if m is None:
        raise ValueError("Could not find 4x4 block in LTA.")
    vals = [float(x) for x in m.group(1).split()]
    if len(vals) < 16:
        raise ValueError("Incomplete 4x4 matrix in LTA.")
    return np.array(vals[:16], dtype=np.float32).reshape(4,4)

# ---- 2) Helpers to go between voxel indices and [0,1] ----
def S_vox_to_01(size_xyz, device=None, dtype=torch.float32):
    X, Y, Z = size_xyz  # XYZ (width,height,depth)
    sx = 1.0/(X-1) if X>1 else 1.0
    sy = 1.0/(Y-1) if Y>1 else 1.0
    sz = 1.0/(Z-1) if Z>1 else 1.0
    S_v2u  = torch.tensor([[sx,0.,0.,0.],
                           [0.,sy,0.,0.],
                           [0.,0.,sz,0.],
                           [0.,0.,0.,1.]], device=device, dtype=dtype)
    S_u2v  = torch.tensor([[1./sx if X>1 else 1.,0.,0.,0.],
                           [0.,1./sy if Y>1 else 1.,0.,0.],
                           [0.,0.,1./sz if Z>1 else 1.,0.],
                           [0.,0.,0.,1.]], device=device, dtype=dtype)
    return S_v2u, S_u2v

# ---- 3) Main: LTA + two NIfTIs -> Theta_01 (fixed01 -> moving01) ----
def lta_to_theta01(
    lta_path: str,
    fixed_nii_path: str,   # the destination (dst) volume (the grid you sample on)
    moving_nii_path: str   # the source (src) volume (the image you pull samples from)
) -> torch.Tensor:
    """
    Returns a 4x4 torch.Tensor that maps fixed [0,1]^3 coords -> moving [0,1]^3 coords.
    Apply to homogeneous coords [x,y,z,1]^T in [0,1].
    """
    # LTA (src RAS -> dst RAS)
    M_ras = torch.from_numpy(read_lta_ras_to_ras(lta_path))

    # Load affines (vox -> RAS) for the volumes you want to use now
    fix_img = nib.load(fixed_nii_path)
    mov_img = nib.load(moving_nii_path)
    A_fix = torch.from_numpy(fix_img.affine.astype(np.float32))
    A_mov = torch.from_numpy(mov_img.affine.astype(np.float32))

    # Build voxel<->unit-cube scalings using XYZ (width,height,depth)
    size_fix_xyz = fix_img.shape[:3]
    size_mov_xyz = mov_img.shape[:3]
    S_fix_v2u, S_fix_u2v = S_vox_to_01(size_fix_xyz, device=M_ras.device, dtype=M_ras.dtype)
    S_mov_v2u, _          = S_vox_to_01(size_mov_xyz, device=M_ras.device, dtype=M_ras.dtype)

    # fixed01 -> moving01
    # x_mov_01 = S_mov_v2u @ A_mov^{-1} @ M_ras @ A_fix @ S_fix_u2v @ x_fix_01
    Theta01 = S_mov_v2u @ torch.linalg.inv(A_mov) @ M_ras @ A_fix @ S_fix_u2v
    return Theta01  # [4,4]

# ---- 4) (Optional) Build and transform a [0,1] meshgrid with Theta01 ----
def meshgrid_01(D, H, W, device=None, dtype=torch.float32):
    z = torch.linspace(0,1,D, device=device, dtype=dtype)
    y = torch.linspace(0,1,H, device=device, dtype=dtype)
    x = torch.linspace(0,1,W, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")  # [D,H,W]
    grid = torch.stack([xx, yy, zz], dim=-1)             # [D,H,W,3]
    ones = torch.ones(D, H, W, 1, device=device, dtype=dtype)
    return torch.cat([grid, ones], dim=-1)               # [D,H,W,4]

def apply_theta01_to_grid(grid01_h: torch.Tensor, Theta01: torch.Tensor) -> torch.Tensor:
    """
    grid01_h : [D,H,W,4] homogeneous [0,1] grid (from meshgrid_01)
    Theta01  : [4,4] mapping fixed01 -> moving01
    Returns  : [D,H,W,3] moving coords in [0,1]
    """
    D,H,W,_ = grid01_h.shape
    moved = grid01_h.view(-1,4) @ Theta01.T              # [DHW,4]
    return moved[:, :3].view(D,H,W,3)

# # ---------- read 4x4 RAS->RAS from .lta ----------
# def read_lta_ras_to_ras(path):
#     """
#     Reads FreeSurfer LTA (type LINEAR_RAS_TO_RAS) and returns a 4x4 RAS->RAS matrix.
#     """
#     with open(path, "r") as f:
#         txt = f.read()
#     block = re.search(r"1\s+4\s+4\s+([\s\S]*?)\nsrc volume info", txt)
#     if block is None:
#         raise ValueError("Could not find the 4x4 block in LTA.")
#     nums = [float(x) for x in block.group(1).split()]
#     if len(nums) < 16:
#         raise ValueError("Not enough numbers for 4x4.")
#     return np.array(nums[:16], dtype=np.float32).reshape(4, 4)

# ---------- normalized [-1,1] helpers ----------
def _norm_from_vox(size_dhw, ac=True):
    D, H, W = size_dhw
    M = torch.eye(4, dtype=torch.float32)
    if ac:
        M[0,0], M[1,1], M[2,2] = 2/(W-1), 2/(H-1), 2/(D-1)
        M[0,3], M[1,3], M[2,3] = -1., -1., -1.
    else:
        M[0,0], M[1,1], M[2,2] = 2/W, 2/H, 2/D
        M[0,3], M[1,3], M[2,3] = -1+1/W, -1+1/H, -1+1/D
    return M

def _vox_from_norm(size_dhw, ac=True):
    D, H, W = size_dhw
    M = torch.eye(4, dtype=torch.float32)
    if ac:
        M[0,0], M[1,1], M[2,2] = (W-1)/2, (H-1)/2, (D-1)/2
        M[0,3], M[1,3], M[2,3] = (W-1)/2, (H-1)/2, (D-1)/2
    else:
        M[0,0], M[1,1], M[2,2] = W/2, H/2, D/2
        M[0,3], M[1,3], M[2,3] = W/2-0.5, H/2-0.5, D/2-0.5
    return M

# # ---------- synthesize vox→RAS for "same FOV, new shape" ----------
def _voxel_sizes_from_affine(A): 
    return np.linalg.norm(A[:3, :3], axis=0)

def _dirs_from_affine(A):
    D = A[:3, :3].copy()
    return D / np.linalg.norm(D, axis=0, keepdims=True)

def make_affine_same_fov(A_old, shape_old_xyz, shape_new_xyz):
    """
    Keep orientation & world origin; adjust voxel size so FOV is unchanged.
    """
    A_new = np.eye(4, dtype=np.float64)
    R = _dirs_from_affine(A_old)                       # 3x3 direction cosines
    s_old = _voxel_sizes_from_affine(A_old)            # old voxel sizes (mm)
    scale = np.array(shape_old_xyz) / np.array(shape_new_xyz)
    s_new = s_old * scale                              # new voxel sizes
    A_new[:3, :3] = R * s_new                          # scale columns
    A_new[:3,  3] = A_old[:3, 3]                       # keep world origin
    return A_new.astype(np.float32)

# # ---------- build theta for RAW tensors on new grids ----------
def theta_for_raw_tensors_from_lta(
    lta_path,
    # Option 1 (recommended): reference NEW nifti headers that define the desired FOV
    ref_mov_new_nii=None, ref_fix_new_nii=None,
    # Option 2: OLD pair used to compute the LTA + synthesize same-FOV affines for new shapes
    ref_mov_old_nii=None, ref_fix_old_nii=None,
    shape_mov_new_xyz=None, shape_fix_new_xyz=None,
    # raw tensor sizes (torch order, DHW = Z,Y,X)
    size_src_dhw=None, size_dst_dhw=None,
    align_corners=True
):
    """
    Returns theta (3x4) mapping fixed_norm -> moving_norm to use with torch.affine_grid on
    RAW torch tensors sized size_src_dhw and size_dst_dhw.
    """
    M_ras = torch.from_numpy(read_lta_ras_to_ras(lta_path))  # 4x4 src_RAS -> dst_RAS

    # Build vox->RAS for the *raw tensor* grids:
    if ref_mov_new_nii and ref_fix_new_nii:
        # Trust the new headers (best case)
        A_src = torch.from_numpy(nib.load(ref_mov_new_nii).affine.astype(np.float32))
        A_dst = torch.from_numpy(nib.load(ref_fix_new_nii).affine.astype(np.float32))
    else:
        # Synthesize same-FOV affines from the OLD pair to the new shapes
        assert (ref_mov_old_nii and ref_fix_old_nii and shape_mov_new_xyz and shape_fix_new_xyz), \
            "Provide old NIfTIs and new XYZ shapes for same-FOV synthesis"
        mov_old = nib.load(ref_mov_old_nii)
        fix_old = nib.load(ref_fix_old_nii)
        A_src = torch.from_numpy(make_affine_same_fov(mov_old.affine, mov_old.shape[:3], shape_mov_new_xyz))
        A_dst = torch.from_numpy(make_affine_same_fov(fix_old.affine, fix_old.shape[:3], shape_fix_new_xyz))

    # Convert RAS->RAS to vox->vox on *your raw tensor* grids
    M_vox = torch.linalg.inv(A_dst) @ M_ras @ A_src  # moving_vox -> fixed_vox
    T_vox = torch.linalg.inv(M_vox)                  # fixed_vox  -> moving_vox

    # Wrap for the RAW tensor sizes (DHW)
    N_in  = _norm_from_vox(size_src_dhw, align_corners)
    V_out = _vox_from_norm(size_dst_dhw, align_corners)
    Theta_4x4 = N_in @ T_vox @ V_out
    return Theta_4x4[:3, :].contiguous()

# ---------- FreeSurfer registration runners ----------
def apply_mri_robust_register(src_nii_path, dst_nii_path, lta_path):
    cmd = [
        "mri_robust_register",
        "--mov", src_nii_path,
        "--dst", dst_nii_path,
        "--lta", lta_path,
        "--iscale", "--satit"
    ]
    subprocess.run(cmd, capture_output=True, text=True)

def register_all_T1s(src_paths, dst_path, lta_paths):
    for i in range(len(src_paths)):
        apply_mri_robust_register(src_paths[i], dst_path, lta_paths[i])
        print(f'finished registered {src_paths[i]}')

# ============================== MAIN ==============================
if __name__ == "__main__":

    src_paths = ["/tcmldrive/NogaK/noga_experiment_data/scan2/t1_mprage_sag_p2_iso_13.nii",
                 "/tcmldrive/NogaK/noga_experiment_data/scan3/t1_mprage_sag_p2_iso_21.nii",
                 "/tcmldrive/NogaK/noga_experiment_data/scan4/t1_mprage_sag_p2_iso_29.nii",
                 "/tcmldrive/NogaK/noga_experiment_data/scan5/t1_mprage_sag_p2_iso_37.nii"]

    lta_paths = ["/tcmldrive/NogaK/noga_experiment_data/scan2/affine_2_to_1.lta",
                 "/tcmldrive/NogaK/noga_experiment_data/scan2/affine_3_to_1.lta",
                 "/tcmldrive/NogaK/noga_experiment_data/scan3/affine_4_to_1.lta",
                 "/tcmldrive/NogaK/noga_experiment_data/scan4/affine_5_to_1.lta"]

    dst_path = "/tcmldrive/NogaK/noga_experiment_data/scan1/t1_mprage_sag_p2_iso_5.nii"

    # run FS registrations (produces the LTA files above)
    # register_all_T1s(src_paths, dst_path, lta_paths)

    # build torch-friendly 4x4 affines for RAW tensors on a new grid
    size_new_dhw = (82, 128, 128)  # DHW for [X,Y,Z]=[128,128,82]
    affines = []
    for i in range(4):
        theta = theta_for_raw_tensors_from_lta(
            lta_path=lta_paths[i],
            ref_mov_new_nii=f"/tcmldrive/NogaK/noga_experiment_data/scan{i+2}/3d_dwi.nii.gz",
            ref_fix_new_nii="/tcmldrive/NogaK/noga_experiment_data/scan1/3d_dwi.nii.gz",
            size_src_dhw=size_new_dhw,
            size_dst_dhw=size_new_dhw,
            align_corners=True
        )
        # Theta01 = lta_to_theta01(lta_paths[i], f"/tcmldrive/NogaK/noga_experiment_data/scan{i+2}/3d_dwi.nii.gz", "/tcmldrive/NogaK/noga_experiment_data/scan1/3d_dwi.nii.gz") 
        affine = torch.eye(4, dtype=torch.float32)
        affine[:-1, :] = theta
        affines.append(affine)

    affines.insert(0, torch.eye(4, dtype=torch.float32))
    # affines = adjust_theta_to_unit_cube(torch.stack(affines), space="minus1_1")
    torch.save(torch.stack(affines), "/tcmldrive/NogaK/noga_experiment_data/affines_for_torch.pt")
