import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nibabel as nib
import sys
sys.path.append('.')
from utils import *

def create_motion_case(path_to_vol, N_stacks, vol_idx, rigid_ranges, voxel_size, return_og_image = False):
    vol = torch.from_numpy(nib.load(path_to_vol).get_fdata()).to(torch.float32)[..., vol_idx]
    D,H,W = vol.shape
    u = torch.rand(rigid_ranges.shape[0])           # [6], uniform in [0,1]
    rigid_params = rigid_ranges[:,0] + (rigid_ranges[:,1]-rigid_ranges[:,0]) * u

    stacks = []
    stacks_indices = []
    affines = []
    for i in range(N_stacks):
        u = torch.rand(rigid_ranges.shape[0])           # [6], uniform in [0,1]
        rigid_params = rigid_ranges[:,0] + (rigid_ranges[:,1]-rigid_ranges[:,0]) * u
        rigid_params = rigid_params.unsqueeze(0) 
        rigid_trans = transformationMatrices(rigid_params[:,:3], rigid_params[:,3:])
        rigid_trans = convert_affine_matrix_from_mm_to_pixels(rigid_trans, voxel_size = voxel_size, image_size = vol.shape)
        grid = F.affine_grid(rigid_trans[:,:-1,:], size=(1,1,D,H,W), align_corners=True)  # [1,D,H,W,3]
        warped = F.grid_sample(vol.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # [1,1,D,H,W]
        stack_idx = torch.range(i,W-1,N_stacks)
        stacks.append(warped[...,stack_idx.tolist()][0,0])
        stacks_indices.append(stack_idx)
        affines.append(rigid_trans.squeeze(0))
    if return_og_image:
        return stacks, stacks_indices, torch.stack(affines), vol
    return stacks, stacks_indices, torch.stack(affines)


class SingleVolSVR_full_simulation(Dataset):
    """
         dataset for volume to volume image registration, each one case for solver solution
    """
    def __init__(self, dwi_nii, N_stacks, vol_idx, rigid_stats, voxel_size, transform = None, return_og_image = False):
  
        super(SingleVolSVR_full_simulation, self).__init__()

        self.dwi_nii = dwi_nii
        self.transform = transform
        self.return_og_image = return_og_image
        self.data = create_motion_case(dwi_nii, N_stacks, vol_idx, rigid_stats, voxel_size, return_og_image=return_og_image)

    
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        if self.return_og_image:
            dwi_stacks, stacks_indices, affines, og_image = self.data  
        else:  
            dwi_stacks, stacks_indices, affines = self.data

        if self.transform:
            dwi_stacks = [self.transform(stack) for stack in dwi_stacks]

        # motion_transformations = get_inverse_transformation(motion_transformations)
        
        output = [dwi_stacks , stacks_indices, affines]
        if self.return_og_image:
            output.append(og_image)

        return output


    

if __name__ == "__main__":
    path_to_vol = '/tcmldrive/NogaK/noga_experiment_data/scan1/ep2d_diff_64dir_iso1.6_s2p2_new_8.nii'

    path_to_mask = '/tcmldrive/NogaK/noga_experiment_data/scan1/T1_mask.nii.gz'
    mask = torch.from_numpy(nib.load(path_to_mask).get_fdata()).to(torch.float32)
    mask[mask!=0] = 1
    bvecs_path = '/tcmldrive/NogaK/noga_experiment_data/scan1/ep2d_diff_64dir_iso1.6_s2p2_new_8.bvec'

    


    rigid_stats = torch.tensor([[-5, 5], #RX - degrees
                                [-5, 5], #RY - degrees
                                [-5, 5], #RZ - degrees
                                [-5, 5], #TX - mm
                                [-5, 5], #TY - mm
                                [-5, 5]])
    voxel_size = torch.tensor([1.758, 1.758, 1.6])

        # Path to your .bvec file (update as needed)
    bvecs_path = '/tcmldrive/NogaK/noga_experiment_data/scan1/ep2d_diff_64dir_iso1.6_s2p2_new_8.bvec'

    save_path = '/tcmldrive/NogaK/noga_experiment_data/og_results'
    # for num in [6,12,24]:

    bvecs_subsets_dir = '/tcmldrive/NogaK/noga_experiment_data/bvecs'

    import os
    # os.mkdir('/tcmldrive/NogaK/noga_experiment_data/full_simulation_ds/corrupted_stack_12')

    idx_file = os.path.join(bvecs_subsets_dir, f"indices_12.txt")


    # Load indices: shape [1, num_of_directions], flatten to (num_of_directions,)
    select_indices = np.loadtxt(idx_file, dtype=int).flatten()

    for idx in select_indices:
        ds = SingleVolSVR_full_simulation(dwi_nii=path_to_vol,
                                            vol_idx=idx, 
                                            N_stacks=5, 
                                            rigid_stats=rigid_stats, 
                                            voxel_size=voxel_size, 
                                            return_og_image = True)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        for i_batch, sample_batched in enumerate(dl):
            dwi_stacks , stacks_indices, affines, og_vol = sample_batched
            for i, stack in enumerate(dwi_stacks):
                save_tensor_as_nii(stack.squeeze(0), f'/tcmldrive/NogaK/noga_experiment_data/full_simulation_ds/corrupted_stack_12/stack_{i+1}_vol_{idx}.nii.gz')

    
    
        



