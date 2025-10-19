import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
import pprint
import sys
sys.path.append('.')
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import transforms, Lambda

from multi_vol_recon.model_recon_multi_vol import ReconNet_canonial_grid
import datetime
import tinycudann as tcnn
from monai.losses.image_dissimilarity import GlobalMutualInformationLoss
from torch.utils.tensorboard import SummaryWriter
from utils import  *
from real_motion_simulations.prepare_experiment_svr import SingleVolSVR

 

class SVR_solver():
    def __init__(self, 
                 path_to_ds, 
                 rigid_config,  
                 recon_config, 
                 lr=0.0001, 
                 full_sim_ds = False,
                 only_canon_grid = False,
                 motion_simulation = True, 
                 vol_to_opt = 0,
                 tensorboard=True,
                 save_exp_path = None,
                 plot_every = None,
                 loss_weights = {'recon_loss' : 1, 'tv_reg' : 0.1, 'dc_loss' : 0.05, 'curv': 0.005},
                 exp_name = '_zero_shot_SVR',
                 device = 'cpu'):
        
        self.only_canon_grid = only_canon_grid
        self.full_sim_ds = full_sim_ds
        # self.input_stacks = input_stacks
        # Load tensorboard
        self.tensorboard = tensorboard
        if tensorboard:
            self.writer = SummaryWriter(comment=exp_name, flush_secs=1)
            self.writer.add_text("Config/rigid", pprint.pformat(rigid_config))
            self.writer.add_text("Config/recon", pprint.pformat(recon_config))
            self.writer.add_text("Config/loss_weights", pprint.pformat(loss_weights))

        # Load the models
        if only_canon_grid:
            self.model = ReconNet_canonial_grid(recon_config, image_size=rigid_config['image_size'], device = device) 
        else:
            self.model = ReconNet(recon_config, image_size=rigid_config['image_size'], device = device)  
        self.rigid_trans_for_recon = torch.load('pred_rigid_trans.pt')[vol_to_opt::13,...].to(device)
 

        self.model.to(device)
        self.model.train()

        # Load the dataset
        to_float32 = Lambda(lambda x: x.type(torch.float32))
        if full_sim_ds:
            self.dataset = SingleVolSVR_full_simulation(dwi_nii=os.path.join(path_to_ds,'scan1/ep2d_diff_64dir_iso1.6_s2p2_new_8.nii'), 
                                                        N_stacks=5, 
                                                        vol_idx=vol_to_opt,
                                                        rigid_stats=rigid_config['ranges'], 
                                                        voxel_size=rigid_config['voxel_size'],                                                        transform=transforms.Compose([ to_float32, ZScoreNormalize3D()] ))
        else:
            self.dataset = SingleVolSVR(experiment_path=path_to_ds,
                                        vol_idx=vol_to_opt,
                                        num_of_grad=5,
                                        slices_to_remove = [0, 81],
                                        transform=transforms.Compose([ to_float32, ZScoreNormalize3D()]), 
                                                    )

        self.loss = nn.MSELoss()
        self.loss_weights = loss_weights
        self.image_size = rigid_config['image_size']

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)



        # training params
        self.min_iter = 200
        self.early_stopping = 50
        self.device = device

        # Saving experiments folder
        if save_exp_path != None:
            self.exp_dir_path = os.path.join(save_exp_path, exp_name + '_' + str(datetime.datetime.now()).replace(' ', '_'))
            os.mkdir(self.exp_dir_path)
            self.plot_every = plot_every
        else:
            self.plot_every = None
            self.exp_dir_path = None


    def forward(self, data, iter):
        if self.only_canon_grid:
            return self.forward_canon_grid(data, iter)
        else:
            return self.forward_trans_grid(data, iter)



    def fit(self, N_iter):
        data = self.dataset[0]  # single case in dataset
        pbar_name = 'slice-to-volume zero-shot registration'
        best_Loss = torch.inf
        iters_without_improvement = 0

        with tqdm.tqdm(desc=pbar_name, total=N_iter, file=sys.stdout) as pbar:
            for iter in range(N_iter):
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True) 
                losses_dict = self.forward(data, iter)
                loss = self.calc_loss(losses_dict)
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                #### print gradient for debug ############################
                # for param in self.model.parameters():
                #     if param.grad is not None:
                #         print(f"Max Gradient: {param.grad.data.max()}")
                #         print(f"Min Gradient: {param.grad.data.min()}")
                #####################################################    
                self.optimizer.step()

                pbar.set_description(f"{pbar_name} (Loss: {loss.item():.3f})")
                pbar.update()

                if self.scheduler:
                    self.scheduler.step(loss.item())

                # look for cons(selfvergence
                if loss.item() < best_Loss:
                    best_Loss = loss.item()
                    iters_without_improvement = 0
                else:
                    if iter > self.min_iter:
                        iters_without_improvement += 1
                        if iters_without_improvement > self.early_stopping:
                            print(f'Stopping training in iter {iter} because there is no improvement in the loss')
                            return self.get_recon_volumes()
                            break

                if self.tensorboard: # write to tensorboard 
                    self.writer.add_scalar(f'loss',  loss, iter)
                    for key, val in losses_dict.items():
                        self.writer.add_scalar(key,  val.detach(), iter)

            return self.get_recon_volumes()

                
    def calc_loss(self, losses):
        final_loss = 0

        for key, val in losses.items():
            final_loss += self.loss_weights[key]*val

        return final_loss.float()
    
    def forward_canon_grid(self, data, iter=-1):

        dwi_stacks , stacks_indices , rigid_trans_for_recon = data

        dwi_stacks = [stack.to(self.device) for stack in dwi_stacks]
        # stack_indices = [idx.to(self.device) for idx in stack_indices]

        warped_stacks, recon_canon, consistency_loss = self.model(self.rigid_trans_for_recon[:,:-1,:], dwi_stacks, stacks_indices)

        # if iter%200 == 0:
        #     save_tensor_as_nii(recon_canon.view(128,128,82), f'debug_plots/training_recon_iter{iter}.nii.gz')
        losses = []
        for i, warped in enumerate(warped_stacks):
            # compare only the acquired slices for this stack
            idx = stacks_indices[i]
            if torch.is_tensor(idx): idx = idx.tolist()

            # assuming stacks are along W (third axis). change indexing if your stacks differ.
            pred = warped[:, :, idx]                 # [H,W,K]
            target = dwi_stacks[i].to(pred.device, pred.dtype)
            losses.append(self.loss(pred,  target))
        
        recon_loss = torch.stack(losses).mean()
        tv_loss = tv3d(recon_canon.unsqueeze(0))
        curv = curvature_loss(recon_canon)
        loss_dict = {'recon_loss' : recon_loss, 'tv_reg':tv_loss, 'dc_loss':consistency_loss, 'curv': curv}
        return loss_dict  

    def forward_trans_grid(self, data, iter=-1):
        dwi_stacks , stacks_indices , rigid_trans_for_recon = data

        dwi_stacks = [stack.to(self.device) for stack in dwi_stacks]
        recon_volumes = self.model(rigid_trans_for_recon[:,:-1,:], dwi_stacks, stacks_indices)


        D,H,W = self.image_size
        losses = []
        # trans_recons = []
        # inv_rigid_trans = get_inverse_transformation(self.rigid_trans_for_recon)
        for i, stack in enumerate(dwi_stacks):

            recon_vol = recon_volumes[i].view( self.image_size)
            losses.append(self.loss(recon_vol[...,stacks_indices[i].tolist()], stack)) 

        recon_loss = torch.stack(losses).mean()
        tv_loss = tv3d(torch.stack(recon_volumes).view(-1,D,H,W))
        loss_dict = {'recon_loss' : recon_loss, 'tv_reg':tv_loss}
        return loss_dict
   
    
    def get_recon_volumes(self):
    
        recons = self.model.get_recons()
        save_tensor_as_nii(recons, f'{self.exp_dir_path}/recon_vol.nii.gz')
        return recons

