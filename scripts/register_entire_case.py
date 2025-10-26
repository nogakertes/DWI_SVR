import sys
sys.path.append('.')
from solver import SVR_solver
import torch

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

    rigid_config = {'ranges' : rigid_stats.to(device), 
                        'add_to_scale' : 2, 
                        'image_size' : torch.tensor([128, 128, 80]).to(device),
                        'voxel_size' : torch.tensor([1.758, 1.758, 1.6]).to(device)}
    
    GNN_args = {'hidden':128, 'out_dim':6, 'heads':16, 'layers':4, 'drop':0.01}
    import itertools
    import numpy as np

    hiddens = [64, 128, 256]
    heads = [4,8,16]
    layers = [4, 6, 8, 10]
    drop = [0, 0.001, 0.01, 0.1]

    # Create all possible combinations for hyperparameter tuning
    hyperparam_combinations = list(itertools.product(hiddens, heads, layers, drop))

    for idx, (hidden, head, layer, drp) in enumerate(hyperparam_combinations):
        print(f"\nRunning GNN hyperparameter tuning experiment {idx+1}/{len(hyperparam_combinations)}")
        
        GNN_args_trial = {
            'hidden': hidden,
            'out_dim': 6,
            'heads': head,
            'layers': layer,
            'drop': drp
        }
        bvec_indices = np.loadtxt(path_to_bvec_indices, dtype=int).tolist()
        exp_name = f"gnn_hyperparam_hidden{hidden}_heads{head}_layers{layer}_drop{drp}"

        try:
            solver = SVR_solver(
                case_path,
                rigid_config,
                GNN_args_trial,
                use_GNN=True,
                only_reg=True,
                only_recon=False,
                input_stacks=False,
                num_of_directions=12,
                predefined_indices=bvec_indices,
                lr=0.001, 
                plot_every=100,
                vol_features_size=64,
                stack_features_size=64,
                save_exp_path='/tcmldrive/NogaK/svr_exps/registration_exps/',
                exp_name=exp_name,
                recon_n_vols=False,
                single_vol_per_epoch_opt=False,
                device=device
            )
            solver.fit(N_iter=1000)
        except Exception as e:
            print(f"Error running experiment {exp_name}: {e}")


    # Load the bvec indices from the text file as a list

    # solver = SVR_solver(
    #     case_path,
    #     rigid_config,
    #     GNN_args,
    #     use_GNN=True,
    #     only_reg=True,
    #     only_recon=False,
    #     input_stacks=False,
    #     num_of_directions=12,
    #     predefined_indices=bvec_indices,
    #     lr=0.001, 
    #     plot_every=100,
    #     vol_features_size=64,
    #     stack_features_size=64,
    #     save_exp_path='/tcmldrive/NogaK/svr_exps/registration_exps/', # do not save during tuning
    #     exp_name='final_exp_reg_new_norm',
    #     recon_n_vols=False,
    #     single_vol_per_epoch_opt=False,
    #     device=device
    # )
    # solver.fit(N_iter=1000)


