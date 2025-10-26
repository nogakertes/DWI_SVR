import sys
sys.path.append('.')
from single_vol_recon.solver_single_vol import SVR_solver
import torch
import os
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
    path_to_bvec_indices = '/tcmldrive/NogaK/noga_experiment_data/bvecs/indices_12.txt'

    bvec_indices = np.loadtxt(path_to_bvec_indices, dtype=int).tolist()

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

    solver = SVR_solver(
        case_path,
        rigid_config,
        recon_config,
        affine_matrix_path = '/tcmldrive/NogaK/svr_exps/registration_exps/gnn_hyperparam_hidden64_heads8_layers4_drop0.001_2025-10-19_16:37:47.767831/pred_rigid_trans.pt',
        only_canon_grid=True,
        full_sim_ds=False,
        lr=0.002,
        plot_every=100,
        vol_to_opt=1,
        vol_indices=bvec_indices,
        loss_weights={'recon_loss': 1, 'tv_reg': 0.05, 'dc_loss': 1, 'curv': 0.05},
        save_exp_path='/tcmldrive/NogaK/svr_exps/',
        exp_name='recon_with_inv_theta_in_dc_loss1_min_coverage2',
        device=device 
            )
    solver.fit(N_iter=5000)


