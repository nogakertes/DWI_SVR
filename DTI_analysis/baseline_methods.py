import numpy as np
import torch
import sys
sys.path.append('.')
from utils import *

def dwi_fit_tensor_least_squares(nifti_path, bvecs_path, bvals_path, method='WLS', norm_imgs = False, mask=None):
    """
    Fits the diffusion tensor using (weighted) least squares to DWI data from .nii.gz, .bvec, .bval files.
    
    Args:
        nifti_path (str): Path to the .nii.gz DWI image.
        bvecs_path (str): Path to the .bvec file.
        bvals_path (str): Path to the .bval file.
        method (str): 'LS' for least squares, 'WLS' for weighted least squares.
        mask (ndarray, optional): Optional binary mask for where to fit the tensor.
    
    Returns:
        dt_tensor (ndarray): The fitted diffusion tensors, shape (..., 6) [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] per voxel.
        fa (ndarray): Fractional anisotropy map.
        md (ndarray): Mean diffusivity map.
    """
    import nibabel as nib
    import dipy.core.gradients
    from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity
    
    # Load DWI data
    img = nib.load(nifti_path)
    data = img.get_fdata()
    # INSERT_YOUR_CODE
    if norm_imgs:
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)
    data[...,0] = data[...,0]*3 
    # Load bvals and bvecs
    bvals = np.loadtxt(bvals_path)
    bvecs = np.loadtxt(bvecs_path)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    gtab = dipy.core.gradients.gradient_table(bvals, bvecs)

    if mask is None:
        # If no mask, default to fitting everywhere with nonzero b=0
        mask = np.ones(data.shape[:3], dtype=bool)

    # Select tensor fitting method
    fit_method = 'WLS' if method.upper() == 'WLS' else 'LS'
    model = TensorModel(gtab, fit_method=fit_method)

    # Fit tensors
    fit = model.fit(data, mask=mask)

    # Extract lower-triangular elements
    dt_tensor = fit.quadratic_form  # shape: (X, Y, Z, 3, 3)
    dt_tensors_6 = fit.lower_triangular()  # shape: (X, Y, Z, 6)
    
    # Compute FA and MD
    fa = fractional_anisotropy(fit.evals)
    md = mean_diffusivity(fit.evals)

    return dt_tensors_6, fa, md



if __name__ == "__main__":
    # import os
    # res_dir = '/tcmldrive/NogaK/noga_experiment_data/og_results'
    # for num in [6,12,24]:
    
    #     bvecs_path = f'/tcmldrive/NogaK/noga_experiment_data/bvecs/bvecs_{num}.bvec'
    #     bvals_path = f'/tcmldrive/NogaK/noga_experiment_data/bvecs/bvals_{num}.bval'
    #     dt_tensors_6, fa, md = dwi_fit_tensor_least_squares(os.path.join(res_dir, f'DWI_{num}.nii.gz'), bvecs_path, bvals_path, 'WLS')
    #     save_tensor_as_nii(torch.tensor(fa), os.path.join(res_dir, f'FA_{num}.nii.gz'))
    num=12
    bvecs_path = f'/tcmldrive/NogaK/noga_experiment_data/bvecs/bvecs_{num}.bvec'
    bvals_path = f'/tcmldrive/NogaK/noga_experiment_data/bvecs/bvals_{num}.bval'
    dt_tensors_6, fa, md = dwi_fit_tensor_least_squares('recons.nii.gz', bvecs_path, bvals_path, method = 'WLS')
    save_tensor_as_nii(torch.tensor(fa), 'recons_fa.nii.gz')