import sys
sys.path.append('.')
from multi_vol_recon.solver_recon_multi_vol import SVR_solver
import torch
import numpy as np

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)

    # Set the random seed for GPU (if using one) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # device:
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    case_path = '/tcmldrive/NogaK/noga_experiment_data/'

    rigid_stats = torch.tensor([[-20, 20], #RX - degrees
                                [-20, 20], #RY - degrees
                                [-20, 20], #RZ - degrees
                                [-20, 20], #TX - mm
                                [-20, 20], #TY - mm
                                [-20, 20]])
    
        
    path_to_bvec_indices = '/tcmldrive/NogaK/noga_experiment_data/bvecs/indices_12.txt'
    bvec_indices = np.loadtxt(path_to_bvec_indices, dtype=int).tolist()

    rigid_config = {'ranges' : rigid_stats.to(device), 
                        'add_to_scale' : 2, 
                        'image_size' : torch.tensor([128, 128, 80]).to(device),
                        'voxel_size' : torch.tensor([1.758, 1.758, 1.6]).to(device)}
    
    
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
                        "n_neurons": 64,
                        "n_hidden_layers": 3
                    }}
 


    #   Load the bvec indices from the text file as a list
    for val in [0.001, 0.01,0.1, 1]:
        solver = SVR_solver(
            case_path,
            recon_config=recon_config,
            rigid_config=rigid_config,
            num_of_directions=12,
            predefined_indices=bvec_indices,
            affine_matrix_path = '/tcmldrive/NogaK/svr_exps/registration_exps/gnn_hyperparam_hidden64_heads8_layers4_drop0.001_2025-10-19_16:37:47.767831/pred_rigid_trans.pt',
            lr=0.002, 
            loss_weights={'recon_loss': 1, 'tv_reg': 0.05, 'dc_loss': 1, 'curv': 0.05, 'MI_loss':0.01},
            save_exp_path='/tcmldrive/NogaK/svr_exps/multi_col_recons/', # do not save during tuning
            exp_name='multi_recon_exp_with_MI_loss_val_'+str(val),
            device=device
        )
        solver.fit(N_iter=1500)


