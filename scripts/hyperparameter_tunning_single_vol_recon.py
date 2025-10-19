import os
import sys
sys.path.append('.')
from single_vol_recon.solver_single_vol import SVR_solver
import torch
from utils import *
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
    
    path_to_bvec_indices = '/tcmldrive/NogaK/noga_experiment_data/bvecs/indices_12.txt'

    bvec_indices = np.loadtxt(path_to_bvec_indices, dtype=int).tolist()


    # Learning Rate Tuning Experiment
    import numpy as np

    # Define the set of learning rates to try
    learning_rates = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]

    # Use a fixed encoding and network config (from above)
    encoding_params = recon_config['encoding']
    network_config = recon_config['network']

    # Fix loss weights for all runs
    fixed_loss_weights = {'recon_loss': 1, 'tv_reg': 0.01, 'dc_loss': 0.1, 'curv': 0.1}

    for i, lr in enumerate(learning_rates):
        try:
            exp_name = f'_SVR_single_vol_fullsim_lrtune_{i}_lr{lr}'
            print(f"Running LR tuning experiment {i+1}/{len(learning_rates)}: {exp_name}")

            recon_config_trial = {
                "encoding": encoding_params,
                "network": network_config
            }

            solver = SVR_solver(
                case_path,
                rigid_config,
                recon_config_trial,
                affine_matrix_path = '/tcmldrive/NogaK/svr_exps/registration_exps/final_exp_reg_GNN_2025-10-19_09:11:26.783973/pred_rigid_trans.pt',
                only_canon_grid=True,
                full_sim_ds=False,
                lr=lr,
                plot_every=100,
                vol_to_opt=1,
                vol_indices=bvec_indices,
                loss_weights=fixed_loss_weights,
                save_exp_path='/tcmldrive/NogaK/svr_exps/lr_hyperparam_exps',
                exp_name=exp_name,
                device=device
            )
            solver.fit(N_iter=5000)
            # Optionally, save results or metrics for analysis

        except Exception as e:
            print(f"Error running LR tuning experiment {i+1}/{len(learning_rates)}: {exp_name}")
            print(e)

