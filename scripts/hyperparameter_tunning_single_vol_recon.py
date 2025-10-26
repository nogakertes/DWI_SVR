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
    import itertools

    # Hyperparameter tuning for loss weights

    # Keep reconstruction loss fixed, tune regularizers
    recon_weight = 1.0
    tv_vals = [0.05]
    dc_vals = [0.1, 1, 10, 100]
    curv_vals = [0.05]

    combos = list(itertools.product(tv_vals, dc_vals, curv_vals))
    total = len(combos)

    # fixed model/training params
    encoding_params = recon_config['encoding']
    network_config = recon_config['network']
    recon_config_trial = {"encoding": encoding_params, "network": network_config}
    lr = 0.001  # keep a reasonable default learning rate for weight tuning

    def fmt_weight(x):
        return str(x).replace('.', 'p')

    for i, (tv, dc, curv) in enumerate(combos):
        try:
            loss_weights = {
                'recon_loss': recon_weight,
                'tv_reg': float(tv),
                'dc_loss': float(dc),
                'curv': float(curv)
            }
            name_suffix = f"tv{fmt_weight(tv)}_dc{fmt_weight(dc)}_curv{fmt_weight(curv)}"
            exp_name = f"_SVR_single_vol_losswt_tune_{i}_{name_suffix}"
            print(f"Running loss-weight tuning {i+1}/{total}: {exp_name} -> {loss_weights}")

            solver = SVR_solver(
                case_path,
                rigid_config,
                recon_config_trial,
                affine_matrix_path='/tcmldrive/NogaK/svr_exps/registration_exps/gnn_hyperparam_hidden64_heads8_layers4_drop0.001_2025-10-19_16:37:47.767831/pred_rigid_trans.pt',
                only_canon_grid=True,
                full_sim_ds=False,
                lr=lr,
                plot_every=100,
                vol_to_opt=1,
                vol_indices=bvec_indices,
                loss_weights=loss_weights,
                save_exp_path='/tcmldrive/NogaK/svr_exps/losses_weights_exps',
                exp_name=exp_name,
                device=device
            )
            solver.fit(N_iter=5000)

        except Exception as e:
            print(f"Error running loss-weight tuning {i+1}/{total}: {exp_name}")
            print(e)
