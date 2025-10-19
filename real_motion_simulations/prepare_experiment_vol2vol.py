import nibabel as nib
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dipy.io import read_bvals_bvecs
import sys
sys.path.append('./.')
from motion_simulations import * 
sys.path.append('/tcmldrive/NogaK/DTI_project/')
from dHCP_utils import get_optimized_gradients_directions
sys.path.append('.')
sys.path.append('/tcmldrive/NogaK/freesurfer/')
import subprocess
from dipy.segment.mask import median_otsu

def load_scan(path_to_scan, num_of_directions = 10):
    # get T1 image and the DWI images
    for file in os.listdir(path_to_scan):
        if file.startswith('t1_mprage_sag_p2_iso') and file.endswith('.nii'):
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
    b0_images = dwi_data[...,bvals==0]
    dw_images = dwi_data[...,bvals!=0]
    dw_bvecs = bvecs[bvals!=0,:]
    dw_bvals = bvals[bvals!=0]

    bvec_indices, _ = get_optimized_gradients_directions(dw_bvecs, num_of_directions)
    dw_filtered_data = np.concatenate([np.expand_dims(b0_images[...,0],-1), dw_images[...,bvec_indices.squeeze()]], axis=-1)
    filtered_bvecs = np.concatenate([np.expand_dims(bvecs[0,...],0) ,dw_bvecs[bvec_indices.squeeze(0),:]])
    filtered_bvals = np.concatenate([np.array([0]) ,dw_bvals[bvec_indices.squeeze(0)]])
    return T1_data, binar_mask, dw_filtered_data, filtered_bvals, filtered_bvecs

def extract_trasform_from_LTA_file(command_type):
    f = open('temp_trans.lta', 'r')
    text = f.readlines()
    if command_type == 'mri_robust_register':
        text_transform = text[8:12]
    elif command_type == 'mri_synthmorph':
        text_transform = text[:4]

    transform = np.array([list(map(float, s.split())) for s in text_transform])
    return transform

def get_motion_params(experiment_path):
    affine_matrices = {}
    scans_file_name = ['scan1' , 'scan2', 'scan3', 'scan4', 'scan5']
    for fixed_scan in scans_file_name:

        fixed_dw_file_name = [name for name in os.listdir(os.path.join(experiment_path, fixed_scan)) if name.startswith('ep2d_diff_64dir_iso1.6_s2p2_new') and name.endswith('.nii')]
        fixed_dw_path = os.path.join(experiment_path, fixed_scan, fixed_dw_file_name[0])

        fixed_T1_file_name = [name for name in os.listdir(os.path.join(experiment_path, fixed_scan)) if name.startswith('t1_mprage_sag_p2_iso_') and name.endswith('.nii')]
        fixed_T1_path = os.path.join(experiment_path, fixed_scan, fixed_T1_file_name[0])

        # Move T1 to DWI space:
        bash_command = f'flirt -in {fixed_T1_path} -ref {fixed_dw_path} -out {os.path.join(experiment_path, fixed_scan, 'T1_resized.nii.gz')}'

        # get brain mask for the T1 image:
        bash_command = f"mri_synthseg --i {os.path.join(experiment_path, fixed_scan, 'T1_resized.nii.gz')} --o {os.path.join(experiment_path, fixed_scan, 'T1_mask.nii.gz')} --cpu"
        process = subprocess.run(bash_command, shell=True, capture_output=True, text=True)



    #     for moving_scan in scans_file_name:

    #         if (fixed_scan, moving_scan) in affine_matrices or fixed_scan == moving_scan or (moving_scan, fixed_scan) in affine_matrices:
    #             continue

    #         moving_T1_file_name = [name for name in os.listdir(os.path.join(experiment_path, moving_scan)) if name.startswith('t1_mprage_sag_p2_iso_') and name.endswith('.nii')]
    #         moving_T1_path = os.path.join(experiment_path, moving_scan, moving_T1_file_name[0])

    #         # run registration:
    #         bash_command = f'mri_robust_register --mov {moving_T1_path} --dst {fixed_T1_path} --lta temp_trans.lta --iscale --satit'
    #         process = subprocess.run(bash_command, shell=True, capture_output=True, text=True)

    #         bash_command = f"mri_convert -at temp_trans.lta {moving_T1_path} {moving_T1_path[:-4]+'registered_to_'+fixed_scan+'.nii.gz'}"
    #         process = subprocess.run(bash_command, shell=True, capture_output=True, text=True)

    #         affine_mat = extract_trasform_from_LTA_file('mri_robust_register')
    #         affine_matrices[(fixed_scan, moving_scan)] = affine_mat
    #         print(f'Finished register {moving_scan} to {fixed_scan}')

    # np.save(os.path.join(experiment_path,'affine_matrices.npy'), affine_matrices, allow_pickle=True)
    # return affine_matrices



def create_motion_case(path_to_scan, num_of_directions = 10):
    numbers = np.arange(0, 11)
    # Shuffle the numbers randomly
    np.random.shuffle(numbers)
    # Split into groups: 4 groups of size 2 and 1 group of size 3
    groups = [numbers[:2], numbers[2:4], numbers[4:6], numbers[6:8], numbers[8:]]
    if os.path.exists(os.path.join(path_to_scan,'affine_matrices.npy')):
        affine_matrices = np.load(os.path.join(path_to_scan,'affine_matrices.npy'), allow_pickle=True).item()
    else:
        affine_matrices = get_motion_params(path_to_scan)
    shuffled_dw_images = []
    shuffled_bvals = []
    shuffled_bvecs = []
    masks = []
    for i in range(1, 6):
        path  = f'{path_to_scan}/scan{i}'
        T1_data, mask, dw_data, bvals, bvecs = load_scan(path, num_of_directions)
        shuffled_dw_images.append(dw_data[...,groups[i-1]])
        shuffled_bvals.append(bvals[groups[i-1]])
        shuffled_bvecs.append(bvecs[groups[i-1],:])
        masks.append(mask)

 
    shuffled_dw_images = np.concatenate(shuffled_dw_images, axis=-1)
    shuffled_bvals = np.concatenate(shuffled_bvals, axis=-1)
    shuffled_bvecs = np.concatenate(shuffled_bvecs, axis=0)

    # prepare the affine matrices for the shuffled data
    b0_index = np.where(shuffled_bvals==0)[0][0]
    b0_scan_num = [i for i, arr in enumerate(groups) if b0_index in arr][0]
    shuffled_masks = np.zeros((num_of_directions+1, mask.shape[0], mask.shape[1], mask.shape[2]))
    shuffled_affine_matrices = np.zeros((num_of_directions+1, 4, 4))
    for i, group in enumerate(groups):
        shuffled_masks[group,...] = masks[i]
        if i == b0_scan_num:
             shuffled_affine_matrices[group, :] = np.eye(4)
        else:
            if (f'scan{b0_scan_num+1}',f'scan{i+1}') in affine_matrices:
                shuffled_affine_matrices[group, :] = affine_matrices[(f'scan{b0_scan_num+1}',f'scan{i+1}')]
                # shuffled_affine_matrices.append(affine_matrices[(f'scan{b0_scan_num+1}',f'scan{i+1}')])
            else:
                inv_affine_mat = get_inverse_transformation(torch.tensor(affine_matrices[(f'scan{i+1}', f'scan{b0_scan_num+1}')]))
                shuffled_affine_matrices[group, :] = inv_affine_mat

    return shuffled_dw_images, shuffled_bvals, shuffled_bvecs, shuffled_affine_matrices, shuffled_masks


class realSimulationDataset(Dataset):
    """
         dataset for volume to volume image registration, each one case for solver solution
    """
    def __init__(self, experiment_path, 
                 num_of_grad = 10,
                 num_simulation = 1,
                 return_tensor = False, 
                 return_mask = False, 
                 transform = None, 
                 svr_dataset = False,
                 only_reg_ds = False
                ):
  
        super(realSimulationDataset, self).__init__()
        self.data = []
        for i in range(num_simulation):
            self.data.append(create_motion_case(experiment_path, num_of_directions = num_of_grad))
        # self.data = create_motion_case(experiment_path, num_of_directions = num_of_grad)
        self.experiment_path = experiment_path
        self.transform = transform
        self.return_tensor = return_tensor
        self.svr_dataset = svr_dataset
        self.return_mask = return_mask
        self.only_reg_ds = only_reg_ds 

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        shuffled_dw_images, shuffled_bvals, shuffled_bvecs, shuffled_affine_matrices, mask = self.data[index]

        images = torch.Tensor(shuffled_dw_images).permute(-1,0,1,2)
        mask = torch.Tensor(mask)
    
        # Load the bvecs
        bvecs = torch.tensor(shuffled_bvecs)
        bvals = torch.tensor(shuffled_bvals)

        if self.transform:
            # b0_image = self.transform(b0_image)
            images = self.transform(images)
        
        for i in range(len(self.transform)):
            if hasattr(self.transform[i], 'padding'):
                    mask = self.transform[i](mask)

        output = [images, bvecs, bvals, mask]

        if self.only_reg_ds:
            b0_idx = np.where(bvals==0)[0].item()
            dw_indices = torch.where(bvals!=0)[0]
            output = [images[b0_idx,...].unsqueeze(0), images[dw_indices,...], torch.tensor(shuffled_affine_matrices[1:,...], dtype=torch.float32), bvecs]

        return output
    



if __name__ == '__main__':

    experiment_path  = '/tcmldrive/NogaK/noga_experiment_data/'
    get_motion_params(experiment_path)
    ds = realSimulationDataset(experiment_path=experiment_path)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    for i_batch, sample_batched in enumerate(dl):
        print('gg')
    # create_motion_case(experiment_path, num_of_directions = 10)

        

