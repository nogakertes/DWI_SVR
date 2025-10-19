import os
import sys
sys.path.append('.')
from utils import *
from real_motion_simulations.prepare_experiment_svr import * 

if __name__ == "__main__":
    ds_path = '/tcmldrive/NogaK/noga_experiment_data'
    path_to_save_data = '/tcmldrive/NogaK/noga_experiment_data/data_for_eddy'
    num_of_directions = 12
    dataset = realSimulationSVRDataset(experiment_path=ds_path, 
                                        return_corrupted_vols = True,
                                        slices_to_remove=[0,81],
                                        num_of_grad=num_of_directions)
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    for i_batch, sample_batched in enumerate(dl):
        _, _, _, _, bvals, mask, _, _, _, dwi_vols, bvecs_vol = sample_batched
        # INSERT_YOUR_CODE

        # Make sure the output directory exists
        os.makedirs(path_to_save_data, exist_ok=True)

        # Prepare filenames (with num_of_directions in the name)
        base_name = f"case_{i_batch}_n{num_of_directions}"

        # Save dwi_vols as .nii.gz
        dwi_nii_path = os.path.join(path_to_save_data, f"{base_name}_dwi.nii.gz")
        mask_nii_path = os.path.join(path_to_save_data, f"{base_name}_mask.nii.gz")

        # dwi_vols: shape (1, H, W, D, directions)
        # Remove batch dim.
        save_tensor_as_nii(dwi_vols.squeeze(0), dwi_nii_path)
        save_tensor_as_nii(mask.squeeze(0), mask_nii_path)


        # Save bvals and bvecs to file
        bvals_path = os.path.join(path_to_save_data, f"{base_name}.bval")
        bvecs_path = os.path.join(path_to_save_data, f"{base_name}.bvec")
        # Remove batch dim.
        bvals_np = bvals.squeeze(0).detach().cpu().numpy()
        bvecs_np = bvecs_vol.squeeze(0).detach().cpu().numpy()

        # Write bvals (single line, space-separated)
        with open(bvals_path, 'w') as f:
            f.write(" ".join(str(float(b)) for b in bvals_np) + "\n")

        # Write bvecs (3 lines: x, y, z, each space-separated)
        # bvecs shape: (num_directions, 3)
        with open(bvecs_path, 'w') as f:
            for j in range(3):
                f.write(" ".join(str(float(v)) for v in bvecs_np[:, j]) + "\n")

        # Prepare index.txt for FSL eddy.
        # According to FSL, index.txt contains for each volume the index into the acqparams.txt.
        # Basic: one acquisition scheme, so all indices are '1'.
        index_txt_path = os.path.join(path_to_save_data, f"{base_name}_index.txt")
        indices = ["1"] * bvals_np.shape[0]
        with open(index_txt_path, 'w') as f:
            f.write(" ".join(indices) + "\n")

        # Prepare axq.txt (acqparams.txt) for FSL eddy.
        # Example: '0 1 0 0.05'  (phase encode in y, 0.05 total readout time)
        # This should be configured as per actual dataset; here's a template:
        axq_txt_path = os.path.join(path_to_save_data, f"{base_name}_acq.txt")
        with open(axq_txt_path, 'w') as f:
            f.write("0 1 0 0.05\n")


        print('g')