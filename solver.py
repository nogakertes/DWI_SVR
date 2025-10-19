import torch
import torch.nn as nn
import tqdm
import os
import pprint
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import transforms, Lambda
# from icon_registration.losses import LNCC
from model import SVRNet, RegNet, RegNet_slice, ReconNet
import datetime
import tinycudann as tcnn
from monai.losses.image_dissimilarity import GlobalMutualInformationLoss
from torch.utils.tensorboard import SummaryWriter
from utils import  *
from real_motion_simulations.prepare_experiment_svr import realSimulationSVRDataset

class SVR_solver():
    def __init__(self, 
                 path_to_ds, 
                 rigid_config,  
                 GNN_args,
                 recon_config = None, 
                 lr=0.0001, 
                 only_reg = False,
                 only_recon = False,
                 input_stacks = False,
                 motion_simulation = True, 
                 num_of_directions = 6, 
                 predefined_indices = None,
                 tensorboard=True,
                 save_exp_path = None,
                 plot_every = None,
                 use_GNN = True,
                 recon_n_vols = True,
                 single_vol_per_epoch_opt = True,
                 slice_emb_for_inr_size = 16,
                 vol_features_size=64, 
                 stack_features_size=64, 
                 loss_weights = {'recon_loss' : 1, 'reg_loss' : 1, 'trans_var_loss' : 0.001},
                 exp_name = '_zero_shot_SVR',
                 device = 'cpu'):


        assert  only_reg and recon_config == None, 'to perfore reconstruction, recon_config must be provided.'
        
        self.motion_simulation = motion_simulation 
        self.only_reg = only_reg
        self.only_recon = only_recon
        self.input_stacks = input_stacks
        if only_recon: 
            self.rigid_trans_for_recon = torch.load('pred_rigid_trans.pt')
        # Load tensorboard
        self.tensorboard = tensorboard
        if tensorboard:
            self.writer = SummaryWriter(comment=exp_name, flush_secs=1)
            self.writer.add_text("Config/rigid", pprint.pformat(rigid_config))
            self.writer.add_text("Config/recon", pprint.pformat(recon_config))
            self.writer.add_text("Config/GNN", pprint.pformat(GNN_args))
            self.writer.add_text("Config/loss_weights", pprint.pformat(loss_weights))

        # Load the models
        if only_reg:
            if self.input_stacks:
                self.model = RegNet(rigid_config = rigid_config,
                                    vol_features_size=vol_features_size, 
                                    stack_features_size=stack_features_size, 
                                    hidden_dims=[128,64,32],
                                    device=device)
            else:
                self.model = RegNet_slice(rigid_config = rigid_config,
                                            GNN_args = GNN_args,
                                            vol_features_size=vol_features_size, 
                                            stack_features_size=stack_features_size , 
                                            use_GNN = use_GNN,
                                            device=device) 
        elif only_recon:
            self.model = ReconNet(recon_config, n_vols=num_of_directions+1, image_size=rigid_config['image_size'], device = device)    
        else:
            self.model = SVRNet(recon_config = recon_config, 
                                rigid_config = rigid_config,
                                GNN_args = GNN_args,
                                vol_features_size=vol_features_size, 
                                stack_features_size=stack_features_size, 
                                slice_emb_for_inr_size = slice_emb_for_inr_size,
                                n_vols=num_of_directions+1,
                                layer_to_slice_emb = -1,
                                recon_n_vols = recon_n_vols,
                                use_GNN = use_GNN,
                                single_vol_per_epoch_opt = single_vol_per_epoch_opt,
                                device=device)
        self.model.to(device)
        self.model.train()

        # Load the dataset
        to_float32 = Lambda(lambda x: x.type(torch.float32))
        self.dataset = realSimulationSVRDataset(experiment_path=path_to_ds, 
                                                transform=transforms.Compose([ to_float32, ZScoreNormalize3D()]),
                                                slices_to_remove=[0,81],
                                                num_of_grad=num_of_directions, 
                                                vol_indices=predefined_indices,
                                                )
        self.num_of_directions = num_of_directions

        self.loss = nn.MSELoss()
        self.loss_weights = loss_weights

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.75, patience=10, verbose=True)



        # training params
        self.min_iter = 400
        self.early_stopping = 20
        self.device = device
        self.single_vol_per_epoch_opt = single_vol_per_epoch_opt

        # Saving experiments folder
        if save_exp_path != None:
            self.exp_dir_path = os.path.join(save_exp_path, exp_name + '_' + str(datetime.datetime.now()).replace(' ', '_'))
            os.mkdir(self.exp_dir_path)
            self.plot_every = plot_every
        else:
            self.plot_every = None
            self.exp_dir_path = None


    def forward(self, data, iter):
        if self.only_reg:
            if self.input_stacks:
                return self.forward_only_reg_stacks(data, iter)
            else:
                return self.forward_only_reg_slices(data, iter)
        elif self.only_recon:
            return self.forward_only_recon(data,iter)

        else:
            return self.forward_with_registrationNrecon(data, iter)



    def fit(self, N_iter):
        data = self.dataset[0]  # single case in dataset
        pbar_name = 'slice-to-volume zero-shot registration'
        best_Loss = torch.inf
        iters_without_improvement = 0

        with tqdm.tqdm(desc=pbar_name, total=N_iter, file=sys.stdout) as pbar:
            for iter in range(N_iter):
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True) 
                if self.only_reg: 
                    losses_dict, pred_rigid_trans_per_stack = self.forward(data, iter)
                else:
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
                            if self.only_reg:
                                torch.save(pred_rigid_trans_per_stack.detach(), os.path.join(self.exp_dir_path, 'pred_rigid_trans.pt'))
                            else:
                                self.get_recon_volumes(data)
                            break

                if self.tensorboard: # write to tensorboard 
                    self.writer.add_scalar(f'loss',  loss, iter)
                    if isinstance(losses_dict, dict):
                        for key, val in losses_dict.items():
                            self.writer.add_scalar(key,  val.detach(), iter)
                    else:
                        self.writer.add_scalar('MI Loss',  loss.detach(), iter)
            if self.only_reg:
                torch.save(pred_rigid_trans_per_stack.detach(),  'pred_rigid_trans.pt')
            else:
                self.get_recon_volumes(data)

                
    def calc_loss(self, losses):
        final_loss = 0
        if isinstance(losses, dict):
            for key, val in losses.items():
                final_loss += self.loss_weights[key]*val
        else:
            final_loss = losses

        return final_loss.float()
            
    def forward_only_reg_slices(self, data, iter):
        if self.motion_simulation:
            # motion_dwi_data, bvecs, bvals, motion_mask_data, T1_image, motion_transformations, stacks_indices = data
            dwi_stacks, dwi_slices, stacks_bvecs, slices_bvecs, bvals, motion_mask_data, T1_image, motion_transformations, stacks_indices = data
        else:
            motion_dwi_data, bvecs, bvals, T1_image, stacks_indices = data

        dwi_stacks = [stack.to(self.device) for stack in dwi_stacks]
        dwi_slices = dwi_slices.to(self.device)
        T1_image = T1_image.to(self.device)
        slices_bvecs = slices_bvecs.to(self.device)
        
        # Forward pass on the registration network to get rigid transformation per stack

        rigid_trans = self.model(dwi_slices, 
                                T1_vol=T1_image,
                                q_space=slices_bvecs,
                                stack_indices=stacks_indices,
                                output_slice_features = False,
                                epoch=iter)
        stack_rigid_trans, MI_losses, var_losses = apply_rigid_trans_slice2stacks(rigid_trans, dwi_stacks, stacks_indices, self.num_of_directions+1, T1_vol = T1_image, calc_reg_MI = True, iter=iter)
        loss_dict = {'reg_loss' : MI_losses, 'trans_var_loss' :var_losses}

        return loss_dict, stack_rigid_trans

    def forward_only_reg_stacks(self, data, iter):
        if self.motion_simulation:
            dwi_stacks, dwi_slices, stacks_bvecs, slices_bvecs, bvals, motion_mask_data, T1_image, motion_transformations, stacks_indices = data
        else:
            motion_dwi_data, bvecs, bvals, T1_image, stacks_indices = data

        motion_dwi_data = [stack.to(self.device) for stack in dwi_stacks]
        T1_image = T1_image.to(self.device)
        bvecs = stacks_bvecs.to(self.device)
        
        # Forward pass on the registration network to get rigid transformation per stack
        rigid_trans = self.model(motion_dwi_data, T1_image, bvecs, stacks_indices, iter)

        # Get transformed stacks
        losses = []
        tran_stacks = []
        for i, stack in enumerate(motion_dwi_data):
            grid = torch.nn.functional.affine_grid(rigid_trans[i,:-1,:].unsqueeze(0), (1, 1 , stack.shape[0],  stack.shape[1],  stack.shape[2]), align_corners=True)
            transformed_stack = torch.nn.functional.grid_sample(stack.unsqueeze(0).unsqueeze(0), grid, align_corners=False)
            N_vol = self.num_of_directions+1
            if i<N_vol: stack_slice_indices = stacks_indices[0]
            if i>=N_vol and i<2*N_vol: stack_slice_indices = stacks_indices[1]
            if i>=2*N_vol and i<3*N_vol: stack_slice_indices = stacks_indices[2]
            if i>=3*N_vol and i<4*N_vol: stack_slice_indices = stacks_indices[3]
            if i>=4*N_vol and i<5*N_vol: stack_slice_indices = stacks_indices[4]
            tran_stacks.append(transformed_stack.detach().squeeze())
            losses.append(1+self.loss(transformed_stack.squeeze(), T1_image[...,stack_slice_indices.tolist()]))
        
        # debug plots
        if iter%20==0:
            import matplotlib.pyplot as plt
            plt.figure()
            fig, axs = plt.subplots(5,7, figsize=(10,10))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(tran_stacks[i][...,10].T.cpu(), cmap='gray')
                ax.set_title(f'MI  = {(1-losses[i]):.3f}')
                ax.axis('off')
            plt.savefig(f'debug_plots/trans_stacks_epoch_{iter}.png')
            plt.close()

        if iter == 0:
            plt.figure()
            fig, axs = plt.subplots(5,7, figsize=(10,10))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(dwi_stacks[i][...,10].T.cpu(), cmap='gray')
                # ax.set_title(f'MI  = {(1-init_losses[i]):.3f}')
                ax.axis('off')
            plt.savefig(f'debug_plots/init_stacks.png')
            plt.close()

        return torch.stack(losses).mean()

   

    def forward_with_registrationNrecon(self, data, iter=-1):
        if self.motion_simulation:
            dwi_stacks, dwi_slices, stacks_bvecs, slices_bvecs, bvals, motion_mask_data, T1_image, motion_transformations, stacks_indices = data
        else:
            motion_dwi_data, bvecs, bvals, T1_image, stacks_indices = data

        dwi_stacks = [stack.to(self.device) for stack in dwi_stacks]
        dwi_slices = dwi_slices.to(self.device)
        stacks_bvecs = stacks_bvecs.to(self.device)
        T1_image = T1_image.to(self.device)
        slices_bvecs = slices_bvecs.to(self.device)

        recon_volumes, reg_MI, trans_var = self.model(dwi_stacks, dwi_slices, T1_image, slices_bvecs, stacks_indices, iter)
        if iter%100 == 0:
            save_tensor_as_nii(recon_volumes[3].view(128,128,-1), f'debug_plots/training_recon_iter{iter}')
        N_vols = self.num_of_directions+1
        if self.single_vol_per_epoch_opt:
            vol_to_opt = self.model.vol_to_opt
            losses = []
            for i, stack in enumerate(dwi_stacks):

                for j in range(len(stacks_indices)):
                    if i >= j * N_vols and i < (j + 1) * N_vols:
                        stack_slice_indices = stacks_indices[j]
                        break
                if i%7 == vol_to_opt:
                    losses.append(self.loss(recon_volumes[...,stack_slice_indices.tolist()], stack)) #some intensity based loss for unsupervised registration.
        else:
            losses = []
            for i, stack in enumerate(dwi_stacks):

                for j in range(len(stacks_indices)):
                    if i >= j * N_vols and i < (j + 1) * N_vols:
                        stack_slice_indices = stacks_indices[j]
                        break

                # losses.append(self.loss(recon_volumes[][...,stack_slice_indices.tolist()], stack)) #some intensity based loss for unsupervised registration.

                losses.append(self.loss(recon_volumes[i].view_as(stack), stack))
        loss_dict = {'recon_loss' : torch.stack(losses).mean(), 'reg_loss' : reg_MI, 'trans_var_loss' :trans_var}

        if iter%20==0:
            import matplotlib.pyplot as plt
            if self.single_vol_per_epoch_opt:
                plt.figure()
                fig, axs = plt.subplots(1,5, figsize=(10,3))
                for i, ax in enumerate(axs):
                        for j in range(len(stacks_indices)):
                            if i >= j * N_vols and i < (j + 1) * N_vols:
                                stack_slice_indices = stacks_indices[j]
                                break
                        if i%7 == self.model.vol_to_opt:
                            ax.imshow(recon_volumes[...,stack_slice_indices.tolist()][...,10].T.cpu().detach(), cmap='gray')
                            ax.set_title(f'vol {self.model.vol_to_opt+1}, MSE  = {(losses[i]):.3f}')
                            ax.axis('off')
                plt.savefig(f'debug_plots/recon_stacks_epoch_{iter}.png')
                plt.close()
            else:
                plt.figure()
                fig, axs = plt.subplots(5,7, figsize=(10,10))
                for i, ax in enumerate(axs.flatten()):
      
                    vol = recon_volumes[i].view(stack.shape[0], stack.shape[1], -1)
                    ax.imshow(vol[...,10].T.cpu().detach(), cmap='gray')
    
                    ax.set_title(f'MSE  = {(losses[i]):.3f}')
                    ax.axis('off')
                plt.savefig(f'debug_plots/recon_stacks_epoch_{iter}.png')
                plt.close()

        return loss_dict
        

    def forward_only_recon(self, data, iter=-1):
        if self.motion_simulation:
            dwi_stacks, dwi_slices, stacks_bvecs, slices_bvecs, bvals, motion_mask_data, T1_image, motion_transformations, stacks_indices = data
        else:
            motion_dwi_data, bvecs, bvals, T1_image, stacks_indices = data

        dwi_stacks = [stack.to(self.device) for stack in dwi_stacks]
        dwi_slices = dwi_slices.to(self.device)
        stacks_bvecs = stacks_bvecs.to(self.device)
        T1_image = T1_image.to(self.device)
        slices_bvecs = slices_bvecs.to(self.device)

        recon_volumes = self.model( self.rigid_trans_for_recon, dwi_stacks, stacks_indices)
        if iter%100 == 0:
            save_tensor_as_nii(recon_volumes[3].view(128,128,-1), f'debug_plots/training_recon_iter{iter}')
        N_vols = self.num_of_directions+1
            
        losses = []
        for i, stack in enumerate(dwi_stacks):
            for j in range(len(stacks_indices)):
                if i >= j * N_vols and i < (j + 1) * N_vols:
                    stack_slice_indices = stacks_indices[j]
                    break
            recon_vol = recon_volumes[i%7].view_as(T1_image)
            losses.append(self.loss(recon_vol[...,stack_slice_indices.tolist()], stack)) 
            # losses.append(self.loss(recon_volumes[i].view_as(stack), stack))
        loss_dict = {'recon_loss' : torch.stack(losses).mean()}

        # if iter%20==0:
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     fig, axs = plt.subplots(5,7, figsize=(10,10))
        #     for i, ax in enumerate(axs.flatten()):
    
        #         vol = recon_volumes[i].view(stack.shape[0], stack.shape[1], -1)
        #         ax.imshow(vol[...,10].T.cpu().detach(), cmap='gray')

        #         ax.set_title(f'MSE  = {(losses[i]):.3f}')
        #         ax.axis('off')
        #     plt.savefig(f'debug_plots/recon_stacks_epoch_{iter}.png')
        #     plt.close()

        return loss_dict

    def get_recon_volumes(self, data):
        
        self.model.requires_grad_(False)
        if self.motion_simulation:
            dwi_stacks, dwi_slices, stacks_bvecs, slices_bvecs, bvals, motion_mask_data, T1_image, motion_transformations, stacks_indices = data
        else:
            motion_dwi_data, bvecs, bvals, T1_image, stacks_indices = data
       
        dwi_stacks = [stack.to(self.device) for stack in dwi_stacks]
        dwi_slices = dwi_slices.to(self.device)
        T1_image = T1_image.to(self.device)
        slices_bvecs = slices_bvecs.to(self.device)

        recons = self.model.get_recons(dwi_stacks ,dwi_slices, T1_image, slices_bvecs, stacks_indices)
        save_tensor_as_nii(torch.stack(recons).permute(1,2,3,0), 'recon_vols.nii.gz')

