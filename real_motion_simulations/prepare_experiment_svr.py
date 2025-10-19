import nibabel as nib
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dipy.io import read_bvals_bvecs
import sys
sys.path.append('.')
from utils import *
import subprocess
from dipy.segment.mask import median_otsu

def load_scan(path_to_scan, num_of_directions = 10, predefined_idx = None):
    # get T1 image and the DWI images
    for file in os.listdir(path_to_scan):
        if file.startswith('T1_resized') and file.endswith('.nii.gz'):
            t1_file_name = file
        if file.startswith('ep2d_diff_64dir_iso1.6_s2p2_new') and file.endswith('.nii'):
            dwi_file_name = file

    T1_obj = nib.load(os.path.join(path_to_scan, t1_file_name))
    T1_data = T1_obj.get_fdata()
    dwi_obj = nib.load(os.path.join(path_to_scan, dwi_file_name))
    dwi_data = dwi_obj.get_fdata()

    mask_obj = nib.load(os.path.join(path_to_scan, 'T1_mask.nii.gz'))
    mask = mask_obj.get_fdata()
    binar_mask = np.zeros_like(mask)
    binar_mask[mask != 0 ]= 1 

    # Get bvals and bvecs
    bval_file, bvec_file = os.path.join(path_to_scan, f'{dwi_file_name[:-4]}.bval'), os.path.join(path_to_scan, f'{dwi_file_name[:-4]}.bvec')
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)

    # filter to get only num_of_directions images
    b0_images = torch.from_numpy(dwi_data[...,bvals==0])
    dw_images = torch.from_numpy(dwi_data[...,bvals!=0])
    dw_bvecs = torch.from_numpy(bvecs[bvals!=0,:])
    dw_bvals = torch.from_numpy(bvals[bvals!=0])
    
    if predefined_idx != None:
        bvec_indices = torch.tensor(predefined_idx)[1:]
    else:
        bvec_indices, _ = select_uniformly_distributed_vectors_on_sphere(dw_bvecs, num_of_directions)
    dw_filtered_data = torch.cat([b0_images[..., 0].unsqueeze(-1), dw_images[..., bvec_indices.squeeze()]], dim=-1)
    filtered_bvecs = torch.cat([torch.from_numpy(bvecs)[0, ...].unsqueeze(0), dw_bvecs[bvec_indices.squeeze(0), :]], dim=0)
    filtered_bvals = torch.cat([torch.tensor([0], dtype=dw_bvals.dtype), dw_bvals[bvec_indices.squeeze(0)]], dim=0)

    json_file_path = os.path.join(path_to_scan, 'ep2d_diff_64dir_iso1.6_s2p2_new_8.json')
    output = [torch.tensor(T1_data, dtype=torch.float32), 
              torch.tensor(binar_mask, dtype=torch.float32), 
              torch.tensor(dw_filtered_data, dtype=torch.float32), 
              torch.tensor(filtered_bvals, dtype=torch.float32), 
              torch.tensor(filtered_bvecs, dtype=torch.float32),
              json_file_path]
    return output

def extract_trasform_from_LTA_file(command_type):
    f = open('temp_trans.lta', 'r')
    text = f.readlines()
    if command_type == 'mri_robust_register':
        text_transform = text[8:12]
    elif command_type == 'mri_synthmorph':
        text_transform = text[:4]

    transform = np.array([list(map(float, s.split())) for s in text_transform])
    return transform

def resize_t1_to_dwi_space(experiment_path):
    scans_file_name = ['scan1' , 'scan2', 'scan3', 'scan4', 'scan5']
    for fixed_scan in scans_file_name:

        fixed_dw_file_name = [name for name in os.listdir(os.path.join(experiment_path, fixed_scan)) if name.startswith('ep2d_diff_64dir_iso1.6_s2p2_new') and name.endswith('.nii')]
        fixed_dw_path = os.path.join(experiment_path, fixed_scan, fixed_dw_file_name[0])

        fixed_T1_file_name = [name for name in os.listdir(os.path.join(experiment_path, fixed_scan)) if name.startswith('t1_mprage_sag_p2_iso_') and name.endswith('.nii')]
        fixed_T1_path = os.path.join(experiment_path, fixed_scan, fixed_T1_file_name[0])

        # Move T1 to DWI space:
        bash_command = f"flirt -in {fixed_T1_path} -ref {fixed_dw_path} -out {os.path.join(experiment_path, fixed_scan, 'T1_resized.nii.gz')}"
        process = subprocess.run(bash_command, shell=True, capture_output=True, text=True)

        # get brain mask for the T1 image:
        bash_command = f"mri_synthseg --i {os.path.join(experiment_path, fixed_scan, 'T1_resized.nii.gz')} --o {os.path.join(experiment_path, fixed_scan, 'T1_mask.nii.gz')} --cpu"
        process = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
    
        bash_command = f"mri_convert -rl {os.path.join(experiment_path, fixed_scan, 'T1_resized.nii.gz')} -rt nearest {os.path.join(experiment_path, fixed_scan, 'T1_mask.nii.gz')} {os.path.join(experiment_path, fixed_scan, 'T1_mask.nii.gz')}"
        process = subprocess.run(bash_command, shell=True, capture_output=True, text=True)

#
def create_motion_case(path_to_scan, num_of_directions=10, return_json_path = False, predefined_idx = None, slices_to_remove = None):
    """
    Create a motion-corrupted DWI data by combining stacks from multiple scans.

    This function synthesizes a motion-affected DWI dataset by selecting stacks
    from multiple individual scans and combining them into a single volume.
    Specifically, it uses 5 stacks per volume, resulting in 
    `num_of_directions * 5` total stacks in the final output.

    *** all the first stacks in all volumes are aligned, and so on with all the stacks

    Parameters:
    ----------
    path_to_scan : str
        Path to the folder containing the DWI scans (named 'scan1' to 'scan5').
        
    num_of_directions : int, optional (default=10)
        Number of diffusion directions to load from each scan.

    Returns:
    -------
    motion_dwi_data : torch.Tensor
        The assembled DWI volume with shape [H, W, D, num_directions], 
        where different stacks are drawn from different scans.

    bvals : np.ndarray
        Array of b-values corresponding to the diffusion directions.

    bvecs : np.ndarray
        Array of b-vectors (diffusion gradients), shape [num_directions, 3].

    motion_transformations : List[torch.Tensor]
        List of affine transformation matrices (4x4) describing the 
        relative motion between the T1 image and the 5 stacks.

    motion_mask_data : torch.Tensor
        Combined binary brain mask volume aligned with the motion_dwi_data.
    """
    if predefined_idx != None:
        num_of_directions = len(predefined_idx)-1 # not including the b0 image
    num_stacks_in_vol = 5

    # Load or compute affine matrices
    affine_path = os.path.join(path_to_scan,'affines_for_torch.pt')
    affine_matrices = torch.load(affine_path)

    # Load scans
    dw_images = []
    masks = []
    for i in range(1, 6):
        path = f'{path_to_scan}/scan{i}'

        T1_data, mask, dw_data, bvals, bvecs, cur_json_file_path = load_scan(path, num_of_directions, predefined_idx = predefined_idx)
        if slices_to_remove != None:
            assert isinstance(slices_to_remove, list), 'if not None, slices_to_remove must be a list of ints'
            slice_idx = [i for i in range(T1_data.shape[-1]) if i not in slices_to_remove]
            T1_data = T1_data[..., slice_idx]
            mask = mask[..., slice_idx]
            dw_data = dw_data[...,slice_idx,:]

        if i == 1:
            T1_image = T1_data  # Placeholder if T1 is needed elsewhere
            json_file_path = cur_json_file_path
        dw_images.append(dw_data)
        masks.append(mask)

    # Initialize motion data structures
    stacks_indices = []
    motion_dwi_data = torch.zeros_like(dw_data)
    motion_mask_data = torch.zeros_like(mask)
    # motion_transformations = [torch.eye(4)]
    
    # Create motion volume by stacking slices from multiple scans
    stack_indices = []

    for j in range(num_stacks_in_vol):
        stack_indices = torch.arange(j, dw_data.shape[2], num_stacks_in_vol)
        stacks_indices.append(stack_indices)
        # for v in range(dw_data.shape[-1]):
        #     rnd_scan = np.random.randint(0, 5)
        #     motion_dwi_data[..., stack_indices, v] = dw_images[rnd_scan][..., stack_indices, v]
        #     motion_mask_data[..., stack_indices] = masks[rnd_scan][..., stack_indices]
        motion_dwi_data[..., stack_indices, :] = dw_images[j][..., stack_indices, :]
        motion_mask_data[..., stack_indices] = masks[j][..., stack_indices]
        # if j < 4:
        #     transform = affine_matrices[('scan1', f'scan{j+2}')]
            # motion_transformations.append(torch.tensor(transform))
    output = [motion_dwi_data, T1_image,  bvals, bvecs, affine_matrices, motion_mask_data, stacks_indices]
    if  return_json_path:
        output.append(json_file_path)
    return output



class realSimulationSVRDataset(Dataset):

    def __init__(self, experiment_path, 
                 num_of_grad = 10,
                 num_simulation = 1,
                 return_tensor = False, 
                 vol_indices = None,
                 return_mask = False, 
                 transform = None, 
                 svr_dataset = False,
                 only_reg_ds = False,
                 return_corrupted_vols = False,
                 return_json_path = False,
                 slices_to_remove = None,
                ):
  
        super(realSimulationSVRDataset, self).__init__()
        self.data = []
        self.data.append(create_motion_case(experiment_path, 
                                                num_of_directions = num_of_grad, 
                                                return_json_path = return_json_path, 
                                                predefined_idx = vol_indices,
                                                slices_to_remove = slices_to_remove))
        self.experiment_path = experiment_path
        self.transform = transform
        self.return_tensor = return_tensor
        self.svr_dataset = svr_dataset
        self.return_mask = return_mask
        self.only_reg_ds = only_reg_ds 
        self.return_corrupted_vols = return_corrupted_vols


    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        dwi_data, T1_image,  bvals, bvecs, motion_transformations, motion_mask_data, stacks_indices = self.data[index]

        # if self.dwi_as_stacks:
        dwi_stacks = []
        for idx in stacks_indices:
            stack_all_vols = dwi_data[...,idx.tolist(), :]
            stacks_list = stack_all_vols.split(1, dim=3)
            stacks_list = [t.squeeze(3) for t in stacks_list]
            dwi_stacks.extend(stacks_list)


        stacks_bvecs = torch.vstack([bvecs]*len(stacks_indices))
        
        dwi_slices = dwi_data.permute(0, 1, 3, 2).reshape(dwi_data.shape[0], dwi_data.shape[1], -1)
        slices_bvecs = bvecs.repeat_interleave(dwi_data.shape[2], dim=0)
         

        if self.transform:
            dwi_stacks = [self.transform(stack) for stack in dwi_stacks]
            dwi_slices = self.transform(dwi_slices.permute(-1,0,1))
            T1_image = self.transform(T1_image)
            dwi_slices[:80,...] = dwi_slices[:80,...]/2.5 

        output = [dwi_stacks, dwi_slices, stacks_bvecs, slices_bvecs, bvals, motion_mask_data, T1_image, motion_transformations, stacks_indices]
        if self.return_corrupted_vols:
            output.append(dwi_data)
            output.append(bvecs)
        return output
    
class SingleVolSVR(Dataset):

    def __init__(self, experiment_path, 
                 num_of_grad = 10,
                 num_simulation = 1,
                 return_tensor = False, 
                 return_mask = False, 
                 transform = None, 
                 vol_indcies_for_case = None,
                 vol_idx = 0,
                 only_reg_ds = False,
                 return_corrupted_vol = False,
                 slices_to_remove = None,
                ):
  
        super(SingleVolSVR, self).__init__()
        self.data = []

        self.data.append(create_motion_case(experiment_path, 
                                            num_of_directions = num_of_grad, 
                                            predefined_idx=vol_indcies_for_case,
                                            slices_to_remove=slices_to_remove))
        self.experiment_path = experiment_path
        self.transform = transform
        self.return_tensor = return_tensor
        self.return_mask = return_mask
        self.only_reg_ds = only_reg_ds 
        self.return_corrupted_vol = return_corrupted_vol
        self.vol_idx = vol_idx


    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        dwi_data, T1_image,  _, _, motion_transformations, motion_mask_data, stacks_indices = self.data[index]

        dwi_data = dwi_data[..., self.vol_idx]        
        dwi_stacks = []

        for idx in stacks_indices:
            stack = dwi_data[...,idx.tolist()]
            dwi_stacks.append(stack)
                 

        if self.transform:
            dwi_stacks = [self.transform(stack) for stack in dwi_stacks]
            T1_image = self.transform(T1_image)

        motion_transformations = get_inverse_transformation(motion_transformations)
        output = [dwi_stacks , stacks_indices, motion_transformations]
        if self.return_corrupted_vol:
            output.append(dwi_data)
        return output


if __name__ == '__main__':

    experiment_path  = '/tcmldrive/NogaK/noga_experiment_data/'
    ds = SingleVolSVR(experiment_path=experiment_path)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    for i_batch, sample_batched in enumerate(dl):
        dwi_stacks , motion_mask_data, T1_image, stacks_indices, motion_transformations = sample_batched
        print('g')
    # create_motion_case(experiment_path, num_of_directions = 10)

        

