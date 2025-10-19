from single_vol_recon.solver_single_vol import SVR_solver
import torch
import os
import sys
sys.path.append('.')
from utils import *
from utils import get_optimized_gradients_directions
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.set_device(3)

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)

    # Set the random seed for GPU (if using one) 
    if torch.cuda.is_available():
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # device:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    case_path = '/tcmldrive/NogaK/noga_experiment_data/'

    rigid_stats = torch.tensor([[-20, 20], #RX - degrees
                                [-20, 20], #RY - degrees
                                [-20, 20], #RZ - degrees
                                [-20, 20], #TX - mm
                                [-20, 20], #TY - mm
                                [-20, 20]])
    
    recon_config = {
                    "encoding": {
                        "otype": "HashGrid",
                        "n_levels": 12,
                        "n_features_per_level": 1,
                        "log2_hashmap_size": 18,
                        "base_resolution": 10 ,
                        "per_level_scale": 1.5,
                        "fixed_point_pos": False
                    },
                    "network": {
                        "otype": "FullyFusedMLP",
                        "activation": "Sine",   
                        "output_activation": "None",
                        "n_neurons": 128,
                        "n_hidden_layers": 6
                    }}

    rigid_config = {'ranges' : rigid_stats, 
                        'add_to_scale' : 2, 
                        'image_size' : torch.tensor([128, 128, 80]),
                        'voxel_size' : torch.tensor([1.758, 1.758, 1.6])}

    # Load bvecs from .bvec file
    import numpy as np

    # Path to your .bvec file (update as needed)
    bvecs_path = '/tcmldrive/NogaK/noga_experiment_data/scan1/ep2d_diff_64dir_iso1.6_s2p2_new_8.bvec'

    num_of_directions = 12
    
    bvecs_subsets_dir = '/tcmldrive/NogaK/noga_experiment_data/bvecs'

    # Get paths for bvecs, bvals, and indices using the number of directions
    bvecs_file = os.path.join(bvecs_subsets_dir, f"bvecs_{num_of_directions}.bvec")
    bvals_file = os.path.join(bvecs_subsets_dir, f"bvals_{num_of_directions}.bval")
    idx_file = os.path.join(bvecs_subsets_dir, f"indices_{num_of_directions}.txt")

    # Load bvecs: shape [3, N]
    bvecs = np.loadtxt(bvecs_file)
    # Load bvals: shape [1, N]
    bvals = np.loadtxt(bvals_file)
    # Load indices: shape [1, num_of_directions], flatten to (num_of_directions,)
    select_indices = np.loadtxt(idx_file, dtype=int).flatten()

    recons = []
    for idx in select_indices:
        exp_name = f'_SVR_single_vol_real_sim_vol_idx_{idx}'
        solver = SVR_solver(
            case_path,
            rigid_config,
            recon_config,
            only_canon_grid=True,
            full_sim_ds=False,
            num_of_directions = 65,
            lr=0.002,
            plot_every=100,
            vol_to_opt=idx.item(),
            loss_weights={'recon_loss': 1, 'tv_reg': 0.03, 'dc_loss': 0.05, 'curv': 0.005},
            save_exp_path='/tcmldrive/NogaK/svr_exps/DW_exps',
            exp_name=exp_name,
            device=device
        )
        recon = solver.fit(N_iter=5000)
        recons.append(recon)
    save_tensor_as_nii(torch.stack(recons).permute(1,2,3,0), 'recons.nii.gz')


