from solver import SVR_solver
import torch

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)

    # Set the random seed for GPU (if using one) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    # device:
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    case_path = '/tcmldrive/NogaK/noga_experiment_data/'

    rigid_stats = torch.tensor([[-20, 20], #RX - degrees
                                [-20, 20], #RY - degrees
                                [-20, 20], #RZ - degrees
                                [-20, 20], #TX - mm
                                [-20, 20], #TY - mm
                                [-20, 20]])
    
    recon_config = {"loss": {"otype": "RelativeL2"
                    },
                    "optimizer": {
                        "otype": "Adam",
                        "learning_rate": 1e-2,
                        "beta1": 0.9,
                        "beta2": 0.99,
                        "epsilon": 1e-15,
                        "l2_reg": 1e-6
                    },
                    "encoding": {
                        "otype": "HashGrid",
                        "n_levels": 16,
                        "n_features_per_level": 1,
                        "log2_hashmap_size": 19,
                        "base_resolution": 12,
                        "per_level_scale": 1.38,
                        "fixed_point_pos": False
                    },
                    "network": {
                        "otype": "FullyFusedMLP",
                        "activation": "Sine",   
                        "output_activation": "None",
                        "n_neurons": 64,
                        "n_hidden_layers": 6
                    }}
    # recon_config = {
    #         "encoding": {
    #     "otype": "Composite",
    #     "nested": [
    #         {
    #             "otype": "HashGrid",
    #             "n_input_dims": 3,          # Spatial coordinates (x, y, z)
    #             "n_dims_to_encode": 3,      # <-- REQUIRED here
    #             "n_levels": 16,
    #             "n_features_per_level": 1,
    #             "log2_hashmap_size": 19,
    #             "base_resolution": 16,
    #             "per_level_scale": 1.5
    #         },
    #         {
    #             "otype": "Identity",
    #             "offset": 0.5,
    #             "n_input_dims":3,
    #             "n_dims_to_encode": 3       # <-- REQUIRED here too if you want explicit split
    #         }

    #     ]}
    #     ,
    #     "network": {
    #         "otype": "FullyFusedMLP",
    #         "activation": "ReLU", # LeakyReLU, SiLU, Exponential, Sine, Sigmoid, ReLU
    #         "output_activation": "None",
    #         "n_neurons": 128,
    #         "n_hidden_layers": 5
    #     }
    # }                  
        

    rigid_config = {'ranges' : rigid_stats.to(device), 
                        'add_to_scale' : 2, 
                        'image_size' : torch.tensor([128, 128, 80]).to(device),
                        'voxel_size' : torch.tensor([1.758, 1.758, 1.6]).to(device)}
    
    GNN_args = {'hidden':128, 'out_dim':6, 'heads':16, 'layers':4, 'drop':0.01}

    solver = SVR_solver(
        case_path,
        rigid_config,
        recon_config,
        GNN_args,
        use_GNN=True,
        only_reg=True,
        only_recon=False,
        input_stacks=False,
        num_of_directions=12,
        lr=0.001, 
        plot_every=100,
        slice_emb_for_inr_size=0,
        vol_features_size=64,
        stack_features_size=64,
        save_exp_path=None, # do not save during tuning
        exp_name='final_exp_reg_GNN',
        recon_n_vols=False,
        single_vol_per_epoch_opt=False,
        device=device
    )
    solver.fit(N_iter=1000)


    # solver = SVR_solver(case_path,
    #                     rigid_config,
    #                     recon_config,
    #                     GNN_args,
    #                     use_GNN = True, 
    #                     only_reg = True,
    #                     only_recon=False,
    #                     input_stacks = False,
    #                     num_of_directions=12,
    #                     lr = 0.0005,
    #                     plot_every=100,
    #                     slice_emb_for_inr_size = 0,
    #                     vol_features_size=64, 
    #                     stack_features_size=64, 
    #                     save_exp_path='/tcmldrive/NogaK/svr_exps',
    #                     exp_name= f'_only_reg_without_inv',
    #                     recon_n_vols = False,
    #                     single_vol_per_epoch_opt = False,
    #                     device=device)
    # solver.fit(N_iter=1000)