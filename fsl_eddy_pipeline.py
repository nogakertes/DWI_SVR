import torch
from real_motion_simulations.prepare_experiment_svr import realSimulationSVRDataset
from torch.utils.data import DataLoader
from utils import *
import os

def bids_pe_to_vec(phase_dir: str):
    """Map BIDS PhaseEncodingDirection (i, i-, j, j-, k, k-) to acqparams vector."""
    m = {
        "i":  ( 1,  0,  0),
        "i-": (-1,  0,  0),
        "j":  ( 0,  1,  0),
        "j-": ( 0, -1,  0),
        "k":  ( 0,  0,  1),
        "k-": ( 0,  0, -1),
    }
    if phase_dir not in m:
        raise ValueError(f"Unknown PhaseEncodingDirection: {phase_dir}")
    return m[phase_dir]

def total_readout_from_json(j):
    """Return TotalReadoutTime in seconds from JSON.
       Prefer 'TotalReadoutTime'; otherwise use (ReconMatrixPE-1)*EffectiveEchoSpacing."""
    if "TotalReadoutTime" in j:
        return float(j["TotalReadoutTime"])
    if "EffectiveEchoSpacing" in j and "ReconMatrixPE" in j:
        esp = float(j["EffectiveEchoSpacing"])        # seconds
        pe  = int(j["ReconMatrixPE"])
        return (pe - 1) * esp
    raise ValueError("Could not find TotalReadoutTime or EffectiveEchoSpacing+ReconMatrixPE.")


def save_bvals_bvecs(bvals_t: torch.Tensor,
                     bvecs_t: torch.Tensor,
                     outdir: str = ".",
                     prefix: str = "dwi"):
    """
    Save PyTorch tensors to FSL-compatible bvals/bvecs files.

    Args:
        bvals_t: shape (N,) or (1,N) or (N,1); b-values in s/mm^2
        bvecs_t: shape (3,N) or (N,3); unit vectors per volume (zeros for b0)
        outdir:  output directory
        prefix: optional prefix for filenames (e.g., 'dwi_')
    """

    # ---- Convert to CPU numpy
    bvals = bvals_t.detach().cpu().double().numpy().squeeze()
    bvecs = bvecs_t.detach().cpu().double().numpy().squeeze()

    # ---- Sanity/shape fixes
    if bvals.ndim != 1:
        raise ValueError(f"bvals must be 1D after squeeze; got shape {bvals.shape}")

    # Make bvecs shape (3, N)
    if bvecs.shape[0] == 3 and bvecs.ndim == 2:
        pass
    elif bvecs.ndim == 2 and bvecs.shape[1] == 3:
        bvecs = bvecs.T  # (N,3) -> (3,N)
    else:
        raise ValueError(f"bvecs must be (3,N) or (N,3); got {bvecs.shape}")

    N = bvals.shape[0]
    if bvecs.shape[1] != N:
        raise ValueError(f"Mismatch: len(bvals)={N} but bvecs has N={bvecs.shape[1]}")

    # ---- Write files (space-separated, as FSL expects)
    bvals_path = os.path.join(outdir, f"{prefix}.bvals")
    bvecs_path = os.path.join(outdir, f"{prefix}.bvecs")

    # bvals: single line of N numbers
    np.savetxt(bvals_path, bvals.reshape(1, -1), fmt="%.6f", delimiter=" ")

    # bvecs: 3 rows (x, y, z), N columns
    np.savetxt(bvecs_path, bvecs, fmt="%.8f", delimiter=" ")

    print(f"Saved:\n  {bvals_path}\n  {bvecs_path}")


def save_motion_corrupted_data_as_nii(dataset_path, save_path = None):
    folder_path = os.path.join(save_path, f"{dataset_path.strip('/').split('/')[-1]}_eddy")
    os.makedirs(folder_path, exist_ok = True)
    # get_motion_params(experiment_path)
    ds = realSimulationSVRDataset(experiment_path=dataset_path, num_of_grad=6, return_corrupted_vols=True, return_json_path = True)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    for i_batch, sample_batched in enumerate(dl):
        dwi_stacks, dwi_slices, stacks_bvecs, slices_bvecs, bvals, motion_mask_data, T1_image, motion_transformations, stacks_indices, corrupted_data, bvecs = sample_batched
        save_tensor_as_nii(corrupted_data[0,...], os.path.join(folder_path, 'dwi.nii.gz'))
        save_bvals_bvecs(bvals, bvecs, outdir=folder_path)
        create_files_for_eddy(ds.json_path, save_path)

def create_files_for_eddy(json_file_path, eddy_files_path):
    # --- read JSON ---
    # dir_path = os.path.dirname(json_file_path)
    acqp_out  = os.path.join(eddy_files_path, "acqparams.txt")
    index_out = os.path.join(eddy_files_path, "index.txt")
    nii_path  = os.path.join(eddy_files_path, f"dwi.nii.gz")  
    with open(json_file_path, "r") as f:
        meta = json.load(f)

    phase_dir = meta["PhaseEncodingDirection"]    # e.g. "j-"
    pe_vec = bids_pe_to_vec(phase_dir)
    trot = total_readout_from_json(meta)

    # --- write acqparams.txt ---
    with open(acqp_out, "w") as f:
        f.write(f"{pe_vec[0]} {pe_vec[1]} {pe_vec[2]} {trot:.7f}\n")
    print(f"Wrote {acqp_out} with line: {pe_vec[0]} {pe_vec[1]} {pe_vec[2]} {trot:.7f}")

    # --- build index.txt (all 1s, one per volume) ---
    img = nib.load(nii_path)
    if img.ndim != 4:
        raise ValueError(f"{nii_path} is not 4D (got shape {img.shape})")
    n_vols = img.shape[3]
    index = np.ones(n_vols, dtype=int)[None, :]  # 1 row, N columns
    np.savetxt(index_out, index, fmt="%d", delimiter=" ")
    print(f"Wrote {index_out} with {n_vols} entries (all '1').")



if __name__ == '__main__':
    save_motion_corrupted_data_as_nii('/tcmldrive/NogaK/noga_experiment_data/','/tcmldrive/NogaK/noga_experiment_data_eddy')
    
