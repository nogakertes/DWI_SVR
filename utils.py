import torch
import torch.nn.functional as F
import numpy as np
import json
import nibabel as nib 
from torch_geometric.nn import knn_graph
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from monai.losses.image_dissimilarity import GlobalMutualInformationLoss
import matplotlib.pyplot as plt

def blur3d_separable(x, sigma=1.0, k=5):
    # x: [1,1,D,H,W]
    import torch.nn.functional as F
    t = torch.arange(k, device=x.device) - k//2
    g = torch.exp(-0.5 * (t.float()/sigma)**2); g = (g/g.sum()).view(1,1,-1)
    x = F.conv3d(x, g[...,None,None], padding=(k//2,0,0))
    x = F.conv3d(x, g[:, :, None, :, None], padding=(0,k//2,0))
    x = F.conv3d(x, g[:, :, None, None, :], padding=(0,0,k//2))
    return x

def tv3d(x: torch.Tensor) -> torch.Tensor:
    """
    Total Variation loss for a batch of 3D volumes.
    Args:
        x: tensor of shape [N, D, H, W]
    Returns:
        scalar tensor, average TV over batch
    """
    dx = (x[:, 1:, :, :] - x[:, :-1, :, :]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dz = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    # return dx + dy + dz
    return dx + dy + dz


import torch
import torch.nn.functional as F
from typing import Tuple, Union

def _parse_spacing(spacing: Union[float, Tuple[float, float, float], None],
                   device, dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns inverse-squared spacings (1/dz^2, 1/dy^2, 1/dx^2) as tensors on the right device/dtype.
    """
    if spacing is None:
        dz = dy = dx = 1.0
    elif isinstance(spacing, (int, float)):
        dz = dy = dx = float(spacing)
    else:
        assert len(spacing) == 3, "spacing must be scalar, None, or a 3-tuple (dz, dy, dx)"
        dz, dy, dx = map(float, spacing)

    inv_dz2 = torch.tensor(1.0 / (dz * dz), device=device, dtype=dtype)
    inv_dy2 = torch.tensor(1.0 / (dy * dy), device=device, dtype=dtype)
    inv_dx2 = torch.tensor(1.0 / (dx * dx), device=device, dtype=dtype)
    return inv_dz2, inv_dy2, inv_dx2


def _laplacian_3d(x: torch.Tensor,
                  spacing: Union[float, Tuple[float, float, float], None] = None) -> torch.Tensor:
    """
    3D Laplacian (6-neighborhood) with anisotropic voxel spacing.

    Args
    ----
    x : torch.Tensor
        [B, D, H, W]  or  [B, C, D, H, W]
    spacing : float | (dz, dy, dx) | None
        Voxel sizes along depth (D=z), height (H=y), width (W=x).
        If None -> isotropic spacing = 1.

    Returns
    -------
    torch.Tensor
        Same shape as input.
    """
    added_channel = False

    if x.dim() == 4:            # [B, D, H, W] -> add channel
        x4d = x.unsqueeze(1)    # [B, 1, D, H, W]
        C = 1
        added_channel = True
    elif x.dim() == 5:          # [B, C, D, H, W]
        x4d = x
        C = x4d.size(1)
    else:
        raise ValueError(f"Expected 4D [B,D,H,W] or 5D [B,C,D,H,W], got {tuple(x.shape)}")

    inv_dz2, inv_dy2, inv_dx2 = _parse_spacing(spacing, x4d.device, x4d.dtype)

    # Discrete anisotropic Laplacian:
    # center = -2*(1/dx^2 + 1/dy^2 + 1/dz^2)
    # neighbors (±x, ±y, ±z) = 1/dx^2, 1/dy^2, 1/dz^2 respectively
    k = x4d.new_zeros((1, 1, 3, 3, 3))
    k[0, 0, 1, 1, 1] = -2.0 * (inv_dx2 + inv_dy2 + inv_dz2)  # center
    # z neighbors (depth, D)
    k[0, 0, 0, 1, 1] =  inv_dz2
    k[0, 0, 2, 1, 1] =  inv_dz2
    # y neighbors (height, H)
    k[0, 0, 1, 0, 1] =  inv_dy2
    k[0, 0, 1, 2, 1] =  inv_dy2
    # x neighbors (width, W)
    k[0, 0, 1, 1, 0] =  inv_dx2
    k[0, 0, 1, 1, 2] =  inv_dx2

    # Depthwise per-channel conv
    weight = k.repeat(C, 1, 1, 1, 1)  # [C,1,3,3,3]
    y = F.conv3d(x4d, weight, padding=1, groups=C)

    return y[:, 0] if added_channel else y


def curvature_loss(volumes: torch.Tensor,
                   spacing: Union[float, Tuple[float, float, float], None] = None,
                   reduction: str = "mean") -> torch.Tensor:
    """
    Squared-Laplacian (curvature) loss for a BATCH of 3D volumes with anisotropic spacing.

    Args
    ----
    volumes : torch.Tensor
        [B, D, H, W]  or  [B, C, D, H, W]
    spacing : float | (dz, dy, dx) | None
        Voxel sizes along (D=z, H=y, W=x). If None -> isotropic spacing = 1.
    reduction : "mean" | "none" | "batch"
        - "mean": single scalar over all dims and batch (default)
        - "batch": per-sample scalar, averaged over channels & voxels -> shape [B]
        - "none": return Laplacian^2 tensor (same shape as input)

    Returns
    -------
    torch.Tensor
        Loss according to `reduction`.
    """
    lap = _laplacian_3d(volumes, spacing=spacing)
    sq = lap * lap

    if reduction == "none":
        return sq

    if reduction == "batch":
        if volumes.dim() == 4:
            B = volumes.size(0)
            return sq.view(B, -1).mean(dim=1)
        else:
            B = volumes.size(0)
            return sq.view(B, -1).mean(dim=1)

    # default: "mean"
    return sq.mean()


def mask_aware_consistency(
    canon_from_stacks: torch.Tensor,   # [N,D,H,W] values in canonical
    theta_grid_c2s: torch.Tensor,      # [N,3,4] canonical->stack (for mask warping)
    stacks_indices: list,              # N lists/tensors of acquired slice indices (along W)
    image_size: tuple,                 # (D,H,W)
    device: str | torch.device,
    *,
    min_coverage: float = 0.5
) -> torch.Tensor:
    """
    Compute mask-aware variance across stacks in canonical space.
    Builds per-stack binary masks in stack space from `stacks_indices`,
    warps them to canonical using `theta_grid_c2s`, then computes a
    coverage-weighted variance in canonical.
    """

    D, H, W = image_size
    N = theta_grid_c2s.shape[0]
    canon_masks = []
    for i in range(N):
        mask_stack = torch.zeros((D, H, W), device=device, dtype=torch.float32)
        idx = stacks_indices[i]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mask_stack[:, :, idx] = 1.0
        mask_stack = mask_stack.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        grid_c = F.affine_grid(theta_grid_c2s[i:i+1], size=(1,1,D,H,W), align_corners=True)
        back_mask = F.grid_sample(mask_stack, grid_c, mode='nearest', padding_mode='zeros', align_corners=True)
        canon_masks.append(back_mask[0,0])

    canon_masks = torch.stack(canon_masks, dim=0)  # [N,D,H,W]
    weight_sum = canon_masks.sum(dim=0) + 1e-8
    weighted_mean = (canon_from_stacks * canon_masks).sum(dim=0) / weight_sum
    weighted_var  = ((canon_from_stacks - weighted_mean.unsqueeze(0)).pow(2) * canon_masks).sum(dim=0) / weight_sum
    valid = (weight_sum > min_coverage).to(weighted_var.dtype)
    return (weighted_var * valid).sum() / (valid.sum() + 1e-8)

def get_inverse_transformation(affine_mat):
    if len(affine_mat.shape) == 2:
        affine_mat = affine_mat.unsqueeze(0)
    if affine_mat.dtype != torch.float32: affine_mat = affine_mat.to(torch.float32)
    translation_vector = affine_mat[:,:-1, 3]*-1
    rotation_mat = torch.linalg.inv(affine_mat[:,:3,:3])
    inv_affine_mat =  affine_mat.clone()
    inv_affine_mat[:,:3,:3] = rotation_mat
    inv_affine_mat[:,:-1, 3] = translation_vector
    return inv_affine_mat

def extract_rigid_parameters(transform_matrix):
    """extract the rotation and translation parameters from the transformation matrix

    Args:
        transform_matrix (torch.tensor): 4X4 affine transformation matrix

    Returns:
        torch.tensors: 3 rotation params and 3 translation params.
    """
    if list(transform_matrix.size()) == [4, 4]:
        transform_matrix = transform_matrix.unsqueeze(0)
    # Extract rotation matrix
    rotation_matrix = transform_matrix[:,:3, :3]

    # Extract translation vector
    translation_vector = transform_matrix[:, :3, 3]

    # Convert rotation matrix to Euler angles (in radians)
    yaw = torch.arctan2(rotation_matrix[:,1, 0], rotation_matrix[:,0, 0]).unsqueeze(0)
    pitch = torch.arctan2(-rotation_matrix[:,2, 0], torch.sqrt(rotation_matrix[:,2, 1]**2 + rotation_matrix[:,2, 2]**2)).unsqueeze(0)
    roll = torch.arctan2(rotation_matrix[:,2, 1], rotation_matrix[:,2, 2]).unsqueeze(0)
    rotations_in_degrees = torch.concat([roll, pitch, yaw]).T*(180/torch.pi)
    # Return the rotation angles and translation parameters
    return rotations_in_degrees, translation_vector

def save_tensor_as_nii(tensor, filename, affine=None):
    """
    Save a PyTorch tensor as a NIfTI (.nii.gz) file.

    Parameters:
        tensor (torch.Tensor): The tensor to save (should be 3D or 4D).
        filename (str): Path to the output .nii.gz file.
        affine (np.ndarray, optional): The affine transformation matrix (default: identity).
    """
    # Convert to NumPy array and ensure it's contiguous
    data = tensor.detach().cpu().numpy()

    # Default to identity affine if none provided
    if affine is None:
        affine = np.eye(4)

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(data, affine, )

    # Save as .nii.gz
    nib.save(nifti_img, filename)


def get_optimized_gradients_directions(bvecs, num_of_directions):
    """ Get from bvecs the closest bvecs to the N optimized diffusion gradient direction from:
    
        Skare, Stefan, et al. "Condition number as a measure of noise performance of diffusion tensor data acquisition schemes with MRI." Journal of magnetic resonance 147.2 (2000): 340-352.

    Args:
        bvecs (torch.tensor): diffusion gradient direction
        num_of_directions(int) : can be only 6,10,20,30,40

    Returns:
        indices(torch.tensor), closest_directions(torch.tensor): indices of optimized bvecs, optimized bvecs
    """
    assert num_of_directions in [6,10,20,30,40], 'can get optimal directions for 6, 10, 20, 30, or 40 directions'
    if isinstance(bvecs, np.ndarray):
        bvecs = torch.tensor(bvecs)
 

    opt_directions_dict = json.load(open('optimal_grads_directions.json'))
    # Fix: Select unique indices for the closest bvecs to the optimized directions.
    # The previous approach could return the same index multiple times if multiple optimized directions
    # are closest to the same bvec. Instead, use a greedy matching to ensure unique selection.

    optimized_directions = torch.tensor(opt_directions_dict[str(num_of_directions)])
    bvecs = bvecs.float()
    optimized_directions = optimized_directions.float()

    # Compute pairwise distances: [num_bvecs, num_opt_dirs]
    distances = torch.cdist(bvecs, optimized_directions, p=2)

    # Greedy matching: for each optimized direction, find the closest bvec, but do not reuse bvecs
    num_bvecs = bvecs.shape[0]
    num_opt = optimized_directions.shape[0]
    selected_indices = []
    used_bvecs = set()

    # For each optimized direction (column), find the closest unused bvec
    for j in range(num_opt):
        # Mask out already used bvecs by setting their distance to inf
        dists = distances[:, j].clone()
        if used_bvecs:
            used_mask = torch.tensor([i in used_bvecs for i in range(num_bvecs)], dtype=torch.bool, device=dists.device)
            dists[used_mask] = float('inf')
        idx = torch.argmin(dists).item()
        selected_indices.append(idx)
        used_bvecs.add(idx)

    indices = torch.tensor(selected_indices).unsqueeze(0)  # shape [1, num_opt]
    closest_directions = bvecs[indices.squeeze(0)]
    return indices, closest_directions

    # optimized_directions = torch.tensor(opt_directions_dict[str(num_of_directions)])
    # distances = torch.nn.functional.pairwise_distance(bvecs.unsqueeze(1), optimized_directions.unsqueeze(0), p=2)  # Calculate pairwise distances

    # _, indices = torch.topk(distances, k=1, dim=0, largest=False)  # Find indices of closest rows in A for each row in B

    # closest_directions = bvecs[indices.squeeze(1)] 
    # return indices, closest_directions

    # from torch_geometric.nn import knn_graph
    # from torch_geometric.utils import add_self_loops
    # from torch_scatter import scatter_add

def rescale_registration_net_output(rigid_params, rigid_params_ranges, add_to_scale):
    scaled_rigid_params = rigid_params_ranges[:,0]*(1+add_to_scale) + torch.sigmoid(rigid_params.view(-1, 6))*(rigid_params_ranges[:,1]-rigid_params_ranges[:,0])*(1+add_to_scale)
    return scaled_rigid_params

class MinMaxNormalize(object):
    """
    Min-Max normalization transform for torch tensors.
    
    Args:
        min_val (float): target minimum value (default: 0.0)
        max_val (float): target maximum value (default: 1.0)
        eps (float): small constant to avoid division by zero
    """
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, eps: float = 1e-8):
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (torch.Tensor): Input tensor of any shape
        Returns:
            torch.Tensor: Min-Max normalized tensor
        """
        t_min, t_max = tensor.min(), tensor.max()
        scale = (t_max - t_min).clamp_min(self.eps)  # avoid division by 0
        tensor_norm = (tensor - t_min) / scale
        return tensor_norm * (self.max_val - self.min_val) + self.min_val

class PercentileNormalize(object):
    """
    Percentile-based normalization for torch tensors.

    Scales values using [p_low, p_high] percentiles:
        x_norm = (clamp(x, p_low, p_high) - p_low) / (p_high - p_low)
        -> mapped to [min_val, max_val]

    Args:
        p_low (float): lower percentile (default: 5.0)
        p_high (float): upper percentile (default: 95.0)
        min_val (float): target min after scaling (default: 0.0)
        max_val (float): target max after scaling (default: 1.0)
        clip (bool): whether to clip to [p_low, p_high] before scaling (default: True)
        eps (float): small constant to avoid division by zero
    """
    def __init__(self,
                 p_low: float = 5.0,
                 p_high: float = 95.0,
                 min_val: float = 0.0,
                 max_val: float = 1.0,
                 clip: bool = True,
                 eps: float = 1e-8):
        assert 0.0 <= p_low < p_high <= 100.0, "Percentiles must satisfy 0 <= p_low < p_high <= 100"
        self.p_low = p_low
        self.p_high = p_high
        self.min_val = min_val
        self.max_val = max_val
        self.clip = clip
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        x = tensor.to(torch.float32)

        # Handle NaNs by computing percentiles on valid entries only
        valid = ~torch.isnan(x)
        if valid.any():
            vals = x[valid]
        else:
            # All NaNs: return tensor filled with min_val
            return torch.full_like(x, fill_value=self.min_val).to(orig_dtype)

        q = torch.tensor([self.p_low / 100.0, self.p_high / 100.0], device=x.device)
        p_low_val, p_high_val = torch.quantile(vals, q)

        # Optional clipping to the robust range
        if self.clip:
            x = x.clamp(min=p_low_val, max=p_high_val)

        # Normalize
        scale = (p_high_val - p_low_val).clamp_min(self.eps)
        x = (x - p_low_val) / scale
        x = x * (self.max_val - self.min_val) + self.min_val

        return x.to(orig_dtype)
    
class ZScoreNormalize3D:
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return tensor  # Avoid division by zero
        return (tensor - mean) / std
    


def transformationMatrices(rotation_params, translation_params, device='cpu'):
    """create N affine matrices 

    Args:
        rotation_params (torch.Tensor): in radians shape of [N,3]
        translation_params (torch.Tensor): shape of [N,3] in millimeters

    Returns:
        mat(torch.Tensor): shape if [N,4,4]
    """
    M_PI = np.pi
    rx , ry, rz = rotation_params[:,0], rotation_params[:,1], rotation_params[:,2]
    tx, ty, tz  = translation_params[:,0], translation_params[:,1], translation_params[:,2]
    cosrx = torch.cos(rx*(M_PI/180.0))
    cosry = torch.cos(ry*(M_PI/180.0))
    cosrz = torch.cos(rz*(M_PI/180.0))
    sinrx = torch.sin(rx*(M_PI/180.0))
    sinry = torch.sin(ry*(M_PI/180.0))
    sinrz = torch.sin(rz*(M_PI/180.0))
    mat = torch.stack([torch.eye(4)] * rotation_params.shape[0]).to(device)
    mat[:,0,0] = cosry*cosrz
    mat[:,0,1] = sinrx*sinry*cosrz-cosrx*sinrz
    mat[:,0,2] = cosrx*sinry*cosrz + sinrx*sinrz
    mat[:,0,3] = tx
    mat[:,1,0] = cosry*sinrz
    mat[:,1,1] = (sinrx*sinry*sinrz + cosrx*cosrz)
    mat[:,1,2] = (cosrx*sinry*sinrz - sinrx*cosrz)
    mat[:,1,3] = ty
    mat[:,2,0] = -sinry
    mat[:,2,1] = sinrx*cosry
    mat[:,2,2] = cosrx*cosry
    mat[:,2,3] = tz
    mat[:,3,3] = 1.0
    return mat


def transform_grid(grid, affines):
    B, D, H, W, _ = grid.shape

    # Step 1: Add a homogeneous 1 to the grid to make it [x, y, z, 1]
    ones = torch.ones((B, D, H, W, 1), device=grid.device)
    grid_hom = torch.cat([grid, ones], dim=-1)          # [B, D, H, W, 4]

    # Step 2: Reshape to apply matrix multiplication
    grid_flat = grid_hom.view(B, -1, 4)                 # [B, N, 4]
    affines = affines.to(grid.device)

    # Step 3: Apply affine transformations (batched)
    grid_transformed = torch.bmm(grid_flat, affines.transpose(1, 2))  # [B, N, 4]

    # Step 4: Remove homogeneous component and reshape back to grid shape
    grid_transformed = grid_transformed[..., :3]                     # [B, N, 3]
    grid_transformed = grid_transformed.view(B, D, H, W, 3) 
    return grid_transformed


import torch

def order_grids_by_volumes(grids, stacks_indices, device):
    """
    Reorganizes 3D grid data into volumetric grids based on the provided stack indices.

    Given a set of 3D grids (each with dimensions [W, H, D, 3]) corresponding to slices or stacks, 
    this function groups them into volumes based on the specified stack indices. Each stack can 
    correspond to a part of a volume, and this function reorders them into their respective 
    volumes.

    Args:
        grids (torch.Tensor): A tensor of shape [n_stacks, W, H, D, 3], where `n_stacks` is the number 
                              of individual stack slices, and `W`, `H`, `D` are the width, height, and 
                              depth of each slice. The last dimension (3) represents features or channels 
                              associated with each voxel.
        stacks_indices (list of lists): A list of lists, where each sublist contains indices specifying 
                                        the slices from the `n_stacks` that should be grouped into volumes. 
                                        The length of the list corresponds to the number of volumes, and each 
                                        sublist contains indices of the stacks that belong to each volume.

    Returns:
        torch.Tensor: A tensor of shape [n_vols, W, H, D, 3], where `n_vols` is the number of volumes 
                      formed, and each volume contains `W`, `H`, and `D` dimensions with 3 channels 
                      corresponding to the stacked slices.

    """
    # Get the dimensions of the grids
    n_stacks, W, H, D, _ = grids.shape


    # Calculate the number of volumes based on stack indices
    n_vols = n_stacks // len(stacks_indices)

    volume_grids = []

    # Loop through volumes and gather the appropriate stacks
    for i in range(n_vols):
        vol_grid = torch.zeros(W, H, D, 3).to(device) # Initialize an empty volume grid
        for j, inds in enumerate(stacks_indices):
            vol_grid[..., inds.tolist(), :] = grids[i + n_vols * j, ..., inds.tolist(), :]
        volume_grids.append(vol_grid)

    # Stack all the volume grids into a single tensor and return
    return torch.stack(volume_grids)

def translation_in_mm_to_pixels(translation_in_mm, voxel_size, image_size, normalized = False):
    """"convert translation parameters scaling from pixels scale to mm 
    Args:
        translation_in_pixels (torch.Tensor or numpy.array): shape of (N, 3)
        voxel_size (torch.Tensor or numpy.array): shape if (1,3)
        image_size (torch.Tensor or numpy.array): shape if (1,3)
        normalized (bool, optional): if True: assume translation_in_pixels is normalized and range between [-1,1](if using torch.nn.affine_grid),
                                     if False: assume the translation_in_pixels is in pixels scale(if simpleITK resampler). Defaults to False.

    Returns:
        torch.Tensor or numpy.array: translation in millimeters
    """
   # voxel_size = torch.Tensor(voxel_size).unsqueeze(1)
    device = translation_in_mm.device
    image_size = torch.Tensor([image_size[0], image_size[1], image_size[2]]).to(device)
    if normalized:
        translation_in_pixels = translation_in_mm*2/(image_size*voxel_size)
    else:
        translation_in_pixels =  translation_in_mm/voxel_size
    return translation_in_pixels

def translation_in_pixels_to_mm(translation_in_pixels, voxel_size, image_size, normalized = False):
    """"convert translation parameters scaling from pixels scale to mm 
    Args:
        translation_in_pixels (torch.Tensor or numpy.array): shape of (N, 3)
        voxel_size (torch.Tensor or numpy.array): shape if (1,3)
        image_size (torch.Tensor or numpy.array): shape if (1,3)
        normalized (bool, optional): if True: assume translation_in_pixels is normalized and range between [-1,1],
                                     if False: assume the translation_in_pixels is in pixels scale. Defaults to False.

    Returns:
        torch.Tensor or numpy.array: translation in millimeters
    """
    if normalized:
        translation_in_mm = translation_in_pixels*image_size*voxel_size/2
    else:
        translation_in_mm = translation_in_pixels*voxel_size
    return translation_in_mm

def convert_affine_matrix_from_mm_to_pixels(transformation_matrices, voxel_size,image_size, normed_pixels = True):
    """convert the translation in the transformation matrices to pixels from mm

    Args:
        transformation_matrices (torch.tensor): shape of (N,4,4), 
        voxel_size (torch.tensor): the size in mm of a voxel, shape = (1,3)
        image_size (torch.tensor): 3D shape of each volume
        normed_pixels (bool, optional): pytorch implementation require normalized pixel between [-1,1] and
                                        simpleIKT require pixels. Defaults to True.
    """
    output_trans_mat = transformation_matrices.clone()
    output_trans_mat[:,:-1,-1] = translation_in_mm_to_pixels(transformation_matrices[:,:-1,-1], 
                                                                    voxel_size, 
                                                                    image_size, 
                                                                    normalized = normed_pixels)
    return output_trans_mat

def convert_affine_matrix_from_pixels_to_mm(transformation_matrices, voxel_size,image_size, normed_pixels = True):
    """convert the translation in the transformation matrices to pixels from mm

    Args:
        transformation_matrices (torch.tensor): shape of (N,4,4), here the translation in pixels(or normed pixels)
        voxel_size (torch.tensor): the size in mm of a voxel, shape = (1,3)
        image_size (torch.tensor): 3D shape of each volume
        normed_pixels (bool, optional): pytorch implementation require normalized pixel between [-1,1] and
                                        simpleIKT require pixels. Defaults to True.
    """    
    output_trans_mat = transformation_matrices.clone()
    output_trans_mat[:,:-1,-1] = translation_in_pixels_to_mm(transformation_matrices[:,:-1,-1], 
                                                                    voxel_size, 
                                                                    image_size, 
                                                                    normalized = normed_pixels)
    return output_trans_mat


def plot_feature_similarity(features, epoch):
    """
    features: tensor of shape [N_stacks, 256]
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    import seaborn as sns
    features = features.detach().cpu().numpy()
    sim_matrix = cosine_similarity(features)
    
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(sim_matrix, cmap='coolwarm')

    # Format the colorbar to show 4 decimal points
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    plt.title(f"Cosine Similarity Between Stacks - Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"debug_plots/similarity_epoch_{epoch}.png")
    plt.close()



def plot_pca_features(features, epoch):
    from sklearn.decomposition import PCA
    features = features.detach().cpu().numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1])    
    plt.title(f"PCA of Feature Vectors - Epoch {epoch}")
    plt.savefig(f"debug_plots/pca_features_epoch_{epoch}.png")
    plt.close()


def build_adjacency_matrix(q_space, slice_indices, slice_thickness, sig_q=0.7, sig_x=0.7):
    stack_map = torch.zeros(82).to(q_space.device)
    for i, vals in enumerate(slice_indices):
        stack_map[vals.tolist()] = i
    stack_map = torch.hstack([stack_map]*7)
    stack_map_A_mat  = (stack_map.unsqueeze(0) == stack_map.unsqueeze(1)).int()
    position = torch.arange(82).to(q_space.device)*slice_thickness
    position = torch.hstack([position]*7)
    dist = torch.cdist(position.unsqueeze(1), position.unsqueeze(1), p=2)/position.max()
    A_x_space = torch.exp(dist/-sig_x)
    A_q_space = torch.exp((1-torch.pow(torch.matmul(q_space, q_space.T),2))/(-sig_q))
    return A_x_space*A_q_space#*stack_map_A_mat



def sparse_a_matrix(q_space, slice_thickness, slice_indices, image_size, k = 4):
    num_slices = image_size[-1].item()
    num_vols = (q_space.shape[0]//image_size[-1]).item()
    stack_map = torch.zeros(num_slices).to(q_space.device)
    for i, vals in enumerate(slice_indices):
        stack_map[vals.tolist()] = i
    stack_map = torch.hstack([stack_map]*num_vols)

    position = torch.arange(num_slices).to(q_space.device)*slice_thickness
    position = torch.hstack([position]*num_vols)
    coords = torch.hstack([q_space, position.unsqueeze(1), stack_map.unsqueeze(1)])
    
    edge_index = knn_graph(coords, k=k, loop=False)  # [2, E]

    # Symmetrize (if needed)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Edge weights: RBF over coordinate distances
    row, col = edge_index
    dist2 = (coords[row] - coords[col]).pow(2).sum(-1)        # [E]
    sigma2 = dist2.median().clamp(min=1e-6)                   # scale
    edge_weight = torch.exp(-dist2 / (sigma2))                # [E], ~[0,1]

    # Add self-loops (weight=1.0)
    edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1.0)

    # GCN normalization: D^{-1/2} A D^{-1/2}
    deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=coords.size(0))
    deg_inv_sqrt = (deg + 1e-12).pow(-0.5)
    norm_weight = deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]

    return edge_index, norm_weight


def apply_rigid_trans_slice2stacks(slice_rigid_trans, dwi_stacks, stacks_indices, N_vols, T1_vol = None, calc_reg_MI = True, iter=-1):
        assert (not calc_reg_MI) or (T1_vol is not None), "T1_vol must be provided if calc_reg_MI is True"
        rigid_trans = slice_rigid_trans.view(N_vols,-1,4,4)
        stack_rigid_trans = []
        stack_rigid_trans_var = []
        reg_MI = []
        init_reg_MI = []
        trans_stacks = []
        MI_func  = GlobalMutualInformationLoss(num_bins=32)  # bins for histogram discretization

        for i, stack in enumerate(dwi_stacks):
            vol_num = i%N_vols
            # print(vol_num)
            # vol_indices = torch.range(i%7, i%7+6*len(stacks_indices), 7).tolist()

            if i<N_vols: stack_slice_indices = stacks_indices[0]
            if i>=N_vols and i<2*N_vols: stack_slice_indices = stacks_indices[1]
            if i>=2*N_vols and i<3*N_vols: stack_slice_indices = stacks_indices[2]
            if i>=3*N_vols and i<4*N_vols: stack_slice_indices = stacks_indices[3]
            if i>=4*N_vols and i<5*N_vols: stack_slice_indices = stacks_indices[4]
            cur_stack_rigid_trans = rigid_trans[vol_num,stack_slice_indices.tolist(),...].mean(0)
            stack_rigid_trans_var.append(rigid_trans[vol_num,stack_slice_indices.tolist(),...].std(0).sum())
            inv_mat = get_inverse_transformation(cur_stack_rigid_trans)
            grid = torch.nn.functional.affine_grid(inv_mat[:,:-1,:], (1, 1 , stack.shape[0],  stack.shape[1],  stack.shape[2]), align_corners=True)

            # grid = torch.nn.functional.affine_grid(cur_stack_rigid_trans[:-1,:].unsqueeze(0), (1, 1 , stack.shape[0],  stack.shape[1],  stack.shape[2]), align_corners=True)
            transformed_stack = torch.nn.functional.grid_sample(stack.unsqueeze(0).unsqueeze(0), grid, align_corners=True)
            trans_stacks.append(transformed_stack.detach().squeeze())
            stack_rigid_trans.append(cur_stack_rigid_trans.detach().squeeze())
            if calc_reg_MI:
                reg_MI.append(1+MI_func(transformed_stack.squeeze(), T1_vol[...,stack_slice_indices.tolist()]))

                if iter==0:
                    init_reg_MI.append(1+MI_func(stack.squeeze(), T1_vol[...,stack_slice_indices.tolist()]))
        if iter==0:
            plt.figure()
            fig, axs = plt.subplots(5,7, figsize=(10,10))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(dwi_stacks[i][...,10].T.cpu(), cmap='gray')
                ax.set_title(f'MI  = {(1-init_reg_MI[i]):.3f}')
                ax.axis('off')
            plt.savefig('debug_plots/trans_stacks_init.png')
            plt.close()
            
        if iter%20==0:
            plt.figure()
            fig, axs = plt.subplots(5,7, figsize=(10,10))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(trans_stacks[i][...,10].T.cpu(), cmap='gray')
                ax.set_title(f'MI  = {(1-reg_MI[i]):.3f}')
                ax.axis('off')
            plt.savefig(f'debug_plots/trans_stacks_epoch_{iter}.png')
            plt.close()
            
        return torch.stack(stack_rigid_trans), torch.stack(reg_MI).mean(), torch.stack(stack_rigid_trans_var).mean()


def get_stack_features_from_slice_features(slices_features, stacks_indices, N_vols, N_stacks):
        slices_features = slices_features.view(N_vols,-1,slices_features.shape[-1])
        stacks_features_list = []
        for i in range(N_stacks):
            vol_num = i%N_vols
            # print(vol_num)
            # vol_indices = torch.range(i%7, i%7+6*len(stacks_indices), 7).tolist()

            if i<N_vols: stack_slice_indices = stacks_indices[0]
            if i>=N_vols and i<2*N_vols: stack_slice_indices = stacks_indices[1]
            if i>=2*N_vols and i<3*N_vols: stack_slice_indices = stacks_indices[2]
            if i>=3*N_vols and i<4*N_vols: stack_slice_indices = stacks_indices[3]
            if i>=4*N_vols and i<5*N_vols: stack_slice_indices = stacks_indices[4]
            stack_features = slices_features[vol_num,stack_slice_indices.tolist(),...].mean(0)
            stacks_features_list.append(stack_features)
        return stacks_features_list

#### from https://github.com/daviddmc/NeSVoR/blob/master/nesvor/utils/psf.py

from typing import List, Tuple, Optional, Callable, Union
import torch
from math import log, sqrt

GAUSSIAN_FWHM = 1 / (2 * sqrt(2 * log(2)))
SINC_FWHM = 1.206709128803223 * GAUSSIAN_FWHM


def resolution2sigma(rx, ry=None, rz=None, /, isotropic=False):
    if isotropic:
        fx = fy = fz = GAUSSIAN_FWHM
    else:
        fx = fy = SINC_FWHM
        fz = GAUSSIAN_FWHM
    assert not ((ry is None) ^ (rz is None))
    if ry is None:
        if isinstance(rx, float) or isinstance(rx, int):
            if isotropic:
                return fx * rx
            else:
                return fx * rx, fy * rx, fz * rx
        elif isinstance(rx, torch.Tensor):
            if isotropic:
                return fx * rx
            else:
                assert rx.shape[-1] == 3
                return rx * torch.tensor([fx, fy, fz], dtype=rx.dtype, device=rx.device)
        elif isinstance(rx, List) or isinstance(rx, Tuple):
            assert len(rx) == 3
            return resolution2sigma(rx[0], rx[1], rx[2], isotropic=isotropic)
        else:
            raise Exception(str(type(rx)))
    else:
        return fx * rx, fy * ry, fz * rz


def get_PSF(
    r_max: Optional[int] = None,
    res_ratio: Tuple[float, float, float] = (1, 1, 3),
    threshold: float = 1e-3,
    device: str = "cpu",
    psf_type: str = "gaussian",
) -> torch.Tensor:
    sigma_x, sigma_y, sigma_z = resolution2sigma(res_ratio, isotropic=False)
    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)
    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=torch.device(device))
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")
    if psf_type == "gaussian":
        psf = torch.exp(
            -0.5
            * (
                grid_x**2 / sigma_x**2
                + grid_y**2 / sigma_y**2
                + grid_z**2 / sigma_z**2
            )
        )
    elif psf_type == "sinc":
        # psf = (
        #     torch.sinc(grid_x / res_ratio[0])
        #     * torch.sinc(grid_y / res_ratio[1])
        #     * torch.exp(-0.5 * grid_z**2 / sigma_z**2)
        # )
        psf = torch.sinc(
            torch.sqrt((grid_x / res_ratio[0]) ** 2 + (grid_y / res_ratio[1]) ** 2)
        ) ** 2 * torch.exp(-0.5 * grid_z**2 / sigma_z**2)
    else:
        raise TypeError(f"Unknown PSF type: <{psf_type}>!")
    psf[psf.abs() < threshold] = 0
    rx = int(torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item())
    ry = int(torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item())
    rz = int(torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item())
    psf = psf[
        rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx
    ].contiguous()
    psf = psf / psf.sum()
    return psf

def apply_psf(volume: torch.Tensor, psf: torch.Tensor, device="cpu") -> torch.Tensor:
    """
    Apply a 3D PSF convolution to a volume using local neighborhoods.
    
    Args:
        volume (torch.Tensor): 3D volume of shape (D, H, W) or (1, 1, D, H, W)
        psf (torch.Tensor): PSF kernel from get_PSF(), shape (kD, kH, kW)
        device (str): Device for computation ("cpu" or "cuda")
        
    Returns:
        torch.Tensor: Blurred volume with the same shape as the input volume.
    """
    # Ensure volume has shape (N, C, D, H, W)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    elif volume.ndim == 4:
        volume = volume.unsqueeze(0)  # (1,1,D,H,W)
    elif volume.ndim != 5:
        raise ValueError("Volume must have shape (D,H,W), (1,D,H,W), or (N,C,D,H,W)")
    
    volume = volume.to(device)
    psf = psf.to(device)

    # Reshape PSF for 3D convolution
    psf = psf.unsqueeze(0).unsqueeze(0)  # Shape: (1,1,kD,kH,kW)

    # Apply 3D convolution (padding = 'same' mode)
    padding = tuple(s // 2 for s in psf.shape[2:])  # symmetric padding
    blurred_volume = torch.nn.functional.conv3d(volume, psf, padding=padding)

    return blurred_volume.squeeze()

def apply_psf_at_points(volume: torch.Tensor,
                        psf: torch.Tensor,
                        points_zyx: torch.Tensor,
                        clamp: bool = True,
                        device: str = "cpu") -> torch.Tensor:
    """
    PSF-weighted sampling at arbitrary voxel-centered points.

    Args
    ----
    volume      : (D,H,W) float tensor
    psf         : (kD,kH,kW) float tensor (normalized or not)
    points_zyx  : (N,3) float/long tensor with voxel indices [z,y,x]
    clamp       : if True, clamp out-of-bounds neighbors to edge
    device      : "cpu" or "cuda"

    Returns
    -------
    values : (N,) tensor of PSF-weighted intensities
    """
    vol = volume.to(device)
    ker = psf.to(device).to(vol.dtype)
    # ker = ker / (ker.sum() + 1e-12)

    D, H, W = vol.shape
    kD, kH, kW = ker.shape
    rD, rH, rW = kD // 2, kH // 2, kW // 2

    # Offsets within the PSF window
    dz, dy, dx = torch.meshgrid(
        torch.arange(-rD, rD + 1, device=device),
        torch.arange(-rH, rH + 1, device=device),
        torch.arange(-rW, rW + 1, device=device),
        indexing="ij"
    )
    offsets = torch.stack([dz, dy, dx], dim=-1).reshape(-1, 3)        # (M,3)
    w = ker.reshape(-1)                                               # (M,)

    # Points → neighbors
    pts = points_zyx.to(device).to(torch.long)                        # (N,3)
    neigh = pts[:, None, :] + offsets[None, :, :]                     # (N,M,3)

    if clamp:
        z = neigh[..., 0].clamp_(0, D - 1)
        y = neigh[..., 1].clamp_(0, H - 1)
        x = neigh[..., 2].clamp_(0, W - 1)
    else:
        # mask OOB neighbors (weight -> 0)
        z, y, x = neigh[..., 0], neigh[..., 1], neigh[..., 2]
        mask = (z >= 0) & (z < D) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
        w = w * mask.float()

        z = z.clamp(0, D - 1); y = y.clamp(0, H - 1); x = x.clamp(0, W - 1)

    z = z.to(torch.long); y = y.to(torch.long); x = x.to(torch.long)

    # Gather neighborhood values and weight
    vals = vol[z, y, x]                                              # (N,M)
    out = (vals * w[None, :]).sum(dim=1)                             # (N,)

    return out
    

import torch.nn.functional as F
from typing import List, Sequence, Tuple, Union


SizeXYZ = Tuple[int, int, int]
SizesXYZ = Union[SizeXYZ, Sequence[SizeXYZ]]

def get_grids_for_batch(
    theta_batch: torch.Tensor,          # [N, 3, 4], fixed_norm -> moving_norm
    size_dst_xyz: SizesXYZ,             # (X,Y,Z) or list/tuple of N (X,Y,Z)
    align_corners: bool = True
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Build sampling grids for a batch of 3x4 thetas.

    Parameters
    ----------
    theta_batch : torch.Tensor
        Shape [N, 3, 4]. Each theta maps fixed_norm -> moving_norm (for affine_grid).
    size_dst_xyz : (X, Y, Z) or sequence of N (X, Y, Z)
        Target output sizes in XYZ order. If a single (X,Y,Z) is provided, it is
        applied to all N items. If a sequence is provided, sizes may differ per item.
    align_corners : bool
        Must match the value used when theta was created and when calling grid_sample.

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        If all sizes identical: a single grid of shape [N, D, H, W, 3].
        If sizes differ: a list of N grids, each of shape [1, D, H, W, 3].
    """
    if theta_batch.ndim != 3 or theta_batch.shape[1:] != (3, 4):
        raise ValueError(f"theta_batch must be [N,3,4], got {tuple(theta_batch.shape)}")

    N = theta_batch.shape[0]

    # Normalize sizes → per-sample list of XYZ tuples
    if isinstance(size_dst_xyz, (list, tuple)) and len(size_dst_xyz) == 3:
        sizes_xyz = [tuple(size_dst_xyz)] * N
    elif isinstance(size_dst_xyz, (list, tuple)) and len(size_dst_xyz) == N:
        sizes_xyz = [tuple(s) for s in size_dst_xyz]
        if any(len(s) != 3 for s in sizes_xyz):
            raise ValueError("Each size in size_dst_xyz must be a 3-tuple (X,Y,Z).")
    else:
        raise ValueError("size_dst_xyz must be (X,Y,Z) or a sequence of N (X,Y,Z) tuples.")

    # Convert XYZ -> DHW for affine_grid
    sizes_dhw = [(z, y, x) for (x, y, z) in sizes_xyz]
    all_same = all(s == sizes_dhw[0] for s in sizes_dhw)

    if all_same:
        D, H, W = sizes_dhw[0]
        # Batched grid: [N, D, H, W, 3]
        return F.affine_grid(theta_batch.to(torch.float32),
                             size=(N, 1, D, H, W),
                             align_corners=align_corners)
    else:
        # Per-sample grids if sizes differ
        grids: List[torch.Tensor] = []
        theta_batch = theta_batch.to(torch.float32)
        for i, (D, H, W) in enumerate(sizes_dhw):
            grid_i = F.affine_grid(theta_batch[i:i+1],
                                   size=(1, 1, D, H, W),
                                   align_corners=align_corners)
            grids.append(grid_i)
        return grids

import torch
import torch.nn.functional as F
from typing import Sequence, Tuple

SizeXYZ = Tuple[int, int, int]  # (X, Y, Z)

def inr_coords_from_theta_batch(
    theta_batch: torch.Tensor,          # [N, 3, 4], fixed_norm -> moving_norm
    size_dst_xyz: SizeXYZ | Sequence[SizeXYZ],
    *,
    align_corners: bool = True,
    to_unit_cube: bool = True,          # True → map [-1,1] → [0,1] for tcnn
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build INR coordinates from a batch of 3x4 thetas.

    Parameters
    ----------
    theta_batch : [N, 3, 4] torch.Tensor
        Each theta maps fixed_norm → moving_norm (for affine_grid).
    size_dst_xyz : (X, Y, Z) or sequence of N (X, Y, Z)
        Target output sizes in XYZ order for each sample. If a single tuple is
        given, it is applied to all N.
    align_corners : bool
        Must match the value used when thetas were created and when sampling.
    to_unit_cube : bool
        If True, map coordinates from [-1,1] to [0,1] (recommended for tcnn HashGrid).
        If False, keep them in [-1,1].
    dtype, device :
        Dtype and device for outputs.

    Returns
    -------
    coords : [N, V, 3] torch.Tensor
        Flattened INR coordinates per sample (V = D*H*W). In [0,1] if
        to_unit_cube=True, else in [-1,1].
    grids  : [N, D, H, W, 3] torch.Tensor
        The dense grids as produced by affine_grid (in [-1,1]).

    Notes
    -----
    - We only return the *coordinate grid*; you can pass `coords[i]` through your
      encoder (e.g., `tcnn.Encoding`) and then the INR MLP for sample i.
    - For XYZ size (X,Y,Z), PyTorch expects DHW=(Z,Y,X) internally.
    """
    if theta_batch.ndim != 3 or theta_batch.shape[1:] != (3, 4):
        raise ValueError(f"theta_batch must be [N,3,4], got {tuple(theta_batch.shape)}")
    N = theta_batch.shape[0]

    # Normalize sizes → per-sample DHW tuples
    if isinstance(size_dst_xyz, (list, tuple)) and len(size_dst_xyz) == 3 and not isinstance(size_dst_xyz[0], (list, tuple)):
        sizes_xyz = [tuple(size_dst_xyz)] * N
    elif isinstance(size_dst_xyz, (list, tuple)) and len(size_dst_xyz) == N:
        sizes_xyz = [tuple(s) for s in size_dst_xyz]
        if any(len(s) != 3 for s in sizes_xyz):
            raise ValueError("Each size must be a 3-tuple (X,Y,Z).")
    else:
        raise ValueError("size_dst_xyz must be (X,Y,Z) or a sequence of N (X,Y,Z) tuples.")

    sizes_dhw = [(z, y, x) for (x, y, z) in sizes_xyz]
    all_same = all(s == sizes_dhw[0] for s in sizes_dhw)

    theta_batch = theta_batch.to(dtype=dtype, device=device)

    if all_same:
        D, H, W = sizes_dhw[0]
        grids = F.affine_grid(theta_batch, size=(N, 1, D, H, W), align_corners=align_corners)  # [N,D,H,W,3]
    else:
        # Different sizes per item → stack per-sample grids
        per = []
        for i, (D, H, W) in enumerate(sizes_dhw):
            g = F.affine_grid(theta_batch[i:i+1], size=(1, 1, D, H, W), align_corners=align_corners)  # [1,D,H,W,3]
            per.append(g)
        grids = torch.cat(per, dim=0)  # [N,D,H,W,3]; note shapes may differ—only valid if all equal!

    # Flatten to [N, V, 3] for INR (V = D*H*W)
    coords = grids.reshape(N, -1, 3).to(dtype=dtype)

    # Optional remap to [0,1] (often required by tinycudann encodings)
    if to_unit_cube:
        coords = (coords + 1.0) * 0.5

    return coords, grids

def select_uniformly_distributed_vectors_on_sphere(vectors: torch.Tensor, subset_size: int) -> torch.Tensor:
    """
    Selects a subset of `subset_size` vectors from `vectors` ([N, 3], assumed normalized)
    such that they are as uniformly spread over the sphere as possible.
    Uses a greedy farthest point sampling (FPS, a.k.a. max-min) approach.
    Does not select the all-zeros vector in the output subset.

    Args:
        vectors (torch.Tensor): Input tensor of shape [N, 3], normalized vectors.
        subset_size (int): Number of vectors to select.

    Returns:
        torch.Tensor: [subset_size, 3] tensor of selected vectors (no all-zeros).
    """
    if subset_size > vectors.size(0):
        raise ValueError(f"subset_size ({subset_size}) cannot be greater than number of input vectors ({vectors.size(0)})")
    device = vectors.device
    N = vectors.size(0)
    zero_mask = ~(vectors.abs().sum(dim=1) < 1e-8)  # True if NOT all zeros

    valid_idxs = torch.where(zero_mask)[0]
    if subset_size > valid_idxs.numel():
        raise ValueError(
            f"Requested subset_size ({subset_size}) is greater than the number of nonzero vectors ({valid_idxs.numel()})."
        )

    # Work only with nonzero vectors for selection
    valid_vectors = vectors[valid_idxs]
    num_valid = valid_vectors.size(0)

    selected_indices_valid = []

    # 1. Start with a random VALID vector
    idx = torch.randint(0, num_valid, (1,), device=device).item()
    selected_indices_valid.append(idx)
    remaining_mask = torch.ones(num_valid, dtype=torch.bool, device=device)
    remaining_mask[idx] = False

    # 2. Iteratively select the next farthest point
    for _ in range(1, subset_size):
        selected = valid_vectors[selected_indices_valid]  # [k,3]
        # Compute cosine similarity (dot product) between all unselected and selected vectors
        # Use absolute values so that vectors pointing in opposite directions are considered similar,
        # thus spreading points evenly across the whole sphere (both hemispheres).
        dots = torch.matmul(valid_vectors, selected.T)  # [num_valid, k]
        dist = 1 - dots.abs()  # use .abs() for true geodesic ("antipodal symmetry") distance
        min_dist, _ = dist.min(dim=1)  # [num_valid], distance to closest selected

        min_dist[~remaining_mask] = -1  # mark already selected to be skipped
        next_idx = torch.argmax(min_dist).item()
        selected_indices_valid.append(next_idx)
        remaining_mask[next_idx] = False

    # Convert selected valid indices back to the original indices
    selected_indices = valid_idxs[torch.tensor(selected_indices_valid, device=device)]
    return selected_indices, vectors[selected_indices]  # [subset_size, 3]

def plot_vectors_on_sphere(vectors, show_axes=True, point_color="red", point_size=20, title="Uniformly Distributed Vectors on Sphere"):
    """
    Plots a set of [N, 3] vectors as points on the unit sphere using an interactive 3D plot.

    Args:
        vectors (torch.Tensor or np.ndarray): Array of shape [N, 3], assumed normalized.
        show_axes (bool): Whether to show coordinate axes.
        point_color (str): Color of plotted points.
        point_size (int): Size of the points.
        title (str): Title of the plot.
    """
    import numpy as np
    import plotly.graph_objs as go

    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
    elif not isinstance(vectors, np.ndarray):
        vectors = np.array(vectors)

    # Create data for the unit sphere wireframe
    sphere_u = np.linspace(0, 2 * np.pi, 60)
    sphere_v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(sphere_u), np.sin(sphere_v))
    y = np.outer(np.sin(sphere_u), np.sin(sphere_v))
    z = np.outer(np.ones_like(sphere_u), np.cos(sphere_v))

    sphere_trace = go.Surface(
        x=x, y=y, z=z,
        opacity=0.2,
        colorscale="Blues",
        showscale=False,
        hoverinfo="skip"
    )

    # Scatter plot for the vectors
    point_trace = go.Scatter3d(
        x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
        mode='markers+text',
        marker=dict(
            size=point_size,
            color=point_color,
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=[str(i) for i in range(len(vectors))],
        textposition="top center",
        name="Selected Vectors"
    )

    plot_data = [sphere_trace, point_trace]

    # Optionally add axes
    if show_axes:
        axes_length = 1.2
        axes_traces = []
        axes_traces.append(go.Scatter3d(
            x=[0, axes_length], y=[0,0], z=[0,0],
            mode='lines',
            line=dict(color='red', width=4),
            name='X'
        ))
        axes_traces.append(go.Scatter3d(
            x=[0, 0], y=[0, axes_length], z=[0,0],
            mode='lines',
            line=dict(color='green', width=4),
            name='Y'
        ))
        axes_traces.append(go.Scatter3d(
            x=[0, 0], y=[0,0], z=[0, axes_length],
            mode='lines',
            line=dict(color='blue', width=4),
            name='Z'
        ))
        plot_data.extend(axes_traces)

    fig = go.Figure(data=plot_data)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2], backgroundcolor='white', showbackground=False, showgrid=False, zeroline=True),
            yaxis=dict(range=[-1.2, 1.2], backgroundcolor='white', showbackground=False, showgrid=False, zeroline=True),
            zaxis=dict(range=[-1.2, 1.2], backgroundcolor='white', showbackground=False, showgrid=False, zeroline=True),
            aspectmode='cube'
        ),
        showlegend=False,
        title=title
    )
    fig.show()

if __name__  == "__main__":
    import os
    bvecs_path = '/tcmldrive/NogaK/noga_experiment_data/scan1/ep2d_diff_64dir_iso1.6_s2p2_new_8.bvec'
    # .bvec files are typically 3 x N (rows: x, y, z; columns: directions)
    bvecs_np = np.loadtxt(bvecs_path)
    if bvecs_np.shape[0] == 3:
        bvecs = torch.from_numpy(bvecs_np.T).float()  # [N, 3]
    else:
        bvecs = torch.from_numpy(bvecs_np).float()    # fall

    save_folder = '/tcmldrive/NogaK/noga_experiment_data/bvecs'
    zero_row_idx = torch.where((bvecs == 0).all(dim=1))[0][0]
    os.makedirs(save_folder, exist_ok=True)
    num_of_directions = [6, 12,24]
    for num in num_of_directions:
        idx, subset = select_uniformly_distributed_vectors_on_sphere(bvecs, num)
        idx = torch.hstack([zero_row_idx, idx])

        cur_bvecs = torch.vstack([torch.zeros(1,3), subset])
        bvals = torch.ones(num+1,)*1000
        bvals[0] = 0
        # Save idx as a txt file
        idx_filename = os.path.join(save_folder, f"indices_{num}.txt")
        np.savetxt(idx_filename, idx.cpu().numpy().reshape(1, -1), fmt="%d")

        # Save bvals
        bvals_filename = os.path.join(save_folder, f"bvals_{num}.bval")
        np.savetxt(bvals_filename, bvals.cpu().numpy().reshape(1, -1), fmt="%d")

        # Save bvecs
        # Format as [3, N] for .bvec: each row is one of x,y,z for all directions
        bvecs_for_save = cur_bvecs.t().cpu().numpy()  # shape [3, N+1]
        bvecs_filename = os.path.join(save_folder, f"bvecs_{num}.bvec")
        np.savetxt(bvecs_filename, bvecs_for_save, fmt="%.8f")
        print(subset)

