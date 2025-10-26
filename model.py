import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, JumpingKnowledge, GraphNorm
import torch.nn.functional as F
from encoders import generate_Resnet3D,ResNet2D, Encoder2D
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
import tinycudann as tcnn
from GNNs import GlobalLocalGNN, GINGNN
from utils import *


class StackPoolingLayer(nn.Module):
    def __init__(self, num_of_stacks=5):
        super(StackPoolingLayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((num_of_stacks, None))  # Adaptive pooling

    def forward(self, x):
        x = self.pool(x) # Apply pooling
        return x

class SliceEncoder(nn.Module):
    def __init__(self, n_features = 100, d_in = 1):
        super(SliceEncoder, self).__init__()
        self.n_features = n_features
        self.SliceEncoder = Encoder2D(d_model=n_features, d_in=d_in)
    
    def forward(self, slices):
        return self.SliceEncoder(slices)
    


class StackEncoder(nn.Module):
    def __init__(self, n_res = 10, n_features = 100, d_in = 1):
        super(StackEncoder, self).__init__()
        self.n_features = n_features
        self.SliceEncoder = Encoder2D(d_model=n_features)#ResNet2D(n_res=n_res, d_model=n_features, d_in=d_in, pretrained=False)
        self.pooling = StackPoolingLayer(1)

    def forward(self, stacks, epoch):
        stacks_features = []
        for stack in stacks:
            slices_features = self.SliceEncoder(stack.permute(-1,0,1).unsqueeze(1))
            stack_features = torch.max(slices_features, dim=0, keepdim=True)[0]#self.pooling(slices_features.unsqueeze(0)).squeeze(0)
            stacks_features.append(stack_features)

        # if epoch%20==0 : plot_feature_similarity(slices_features.squeeze(1), epoch) # debugging...

        return torch.stack(stacks_features)
    
class EdgeGNNLayer(MessagePassing):
    def __init__(self, in_channels, edge_dim, out_channels):
        super().__init__(aggr='mean')  # or "add", "mean"
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        # Learnable transformation for edge attributes
        self.edge_attr_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),  # Edge attribute transformation
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # Ensure self loops (optional)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # If edge_attr does not include self-loop edge values, pad with zeros
        if edge_attr.size(0) != edge_index.size(1):
            device = edge_attr.device
            self_loop_attr = torch.zeros(edge_index.size(1) - edge_attr.size(0), edge_attr.size(1)).to(device)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        # Update edge attributes through the learnable edge_attr_mlp
        edge_attr = self.edge_attr_mlp(edge_attr)  # Update edge attributes

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: source node features [num_edges, in_channels]
        # edge_attr: [num_edges, edge_dim]
        msg_input = torch.cat([x_j, edge_attr], dim=1)
        return self.message_mlp(msg_input)
    

class regGNN(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_dims=[128, 64, 32], out_dim=6, num_layers=3):
        super().__init__()
        assert num_layers == len(hidden_dims), "Number of layers not equal to hidden_dims"
        assert num_layers >= 2, "Number of layers should be at least 2"

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(EdgeGNNLayer(in_channels, edge_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(EdgeGNNLayer(hidden_dims[i], edge_dim, hidden_dims[i + 1]))

        # Output layer
        self.layers.append(EdgeGNNLayer(hidden_dims[-1], edge_dim, out_dim))

    def forward(self, x, edge_index, edge_attr):
        # Pass through all layers, updating the edge attributes at each step
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.layers) - 1:
                x = torch.relu(x)  # Apply ReLU only for hidden layers

            # Update edge_attr after each layer (for edge transformation)
            edge_attr = layer.edge_attr_mlp(edge_attr)  # Apply transformation to edge_attr at each layer

        return x  
    

class AttnRegGNN(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden=128, out_dim=6, heads=4, layers=3, drop=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # first layer
        self.convs.append(TransformerConv(in_channels, hidden // heads,
                                          heads=heads, edge_dim=edge_dim, dropout=drop, beta=True))
        self.norms.append(GraphNorm(hidden))

        # hidden layers
        for _ in range(layers - 1):
            self.convs.append(TransformerConv(hidden, hidden // heads,
                                              heads=heads, edge_dim=edge_dim, dropout=drop, beta=True))
            self.norms.append(GraphNorm(hidden))

        self.jk = JumpingKnowledge(mode='cat')  # or 'max' / 'last'
        self.head = nn.Sequential(
            nn.Linear(hidden * layers, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        outs = []

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)     # [N, hidden]
            h = norm(h)
            h = torch.relu(h)
            h = h + x if h.shape == x.shape else h # residual (first layer skips if dims differ)
            outs.append(h)
            x = h
        x = self.jk(outs)         
        out = self.head(x)          # [N, hidden*layers]
        return out[:,:6], out[:,6:]


class SliceFeaturesEncoder(nn.Module):
    def __init__(self, input_dim=128, output_dim=6):
        super(SliceFeaturesEncoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)         # [B, 6]
        )

    def forward(self, x):
        return self.fc(x)

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)  # or nn.init.kaiming_uniform_(layer.weight) for He initialization
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

class RegNet(nn.Module):
    def __init__(self, rigid_config, vol_features_size=128, stack_features_size=128, hidden_dims=[128,64,32], device='cpu', dtype = torch.float32):
        super(RegNet, self).__init__()
        self.VolumeEncoder = generate_Resnet3D(model_depth=18, n_classes=vol_features_size).to(dtype)
        self.StackEncoder = StackEncoder(n_res = 10, n_features = stack_features_size, d_in = 1).to(dtype)
        self.GNN = regGNN(in_channels=stack_features_size+vol_features_size, edge_dim=1, hidden_dims=hidden_dims, num_layers=len(hidden_dims)).to(dtype)
        # self.GNN.apply(initialize_weights)
        self.registration_head = SliceFeaturesEncoder(input_dim=stack_features_size+vol_features_size)
        self.rigid_parameters_ranges = rigid_config['ranges']
        self.add_to_scale = rigid_config['add_to_scale']
        self.voxel_size = rigid_config['voxel_size']
        self.image_size = rigid_config['image_size']
        self.device = device

    def forward(self, dwi_stacks, T1_vol, q_space, stack_indices, epoch=-1):

        # Extract the features vector for each stack of slices and T1 vol
        vol_features = self.VolumeEncoder(T1_vol.unsqueeze(0).unsqueeze(0))
        stacks_features = self.StackEncoder(dwi_stacks, epoch)
        com_features = torch.concat([stacks_features, vol_features.unsqueeze(1).repeat(stacks_features.shape[0], 1, 1)], dim=-1)
        if epoch%20==0 : plot_feature_similarity(stacks_features.squeeze(1), epoch)

        # Extract Edges values
        # edges_weights = build_adjacency_matrix(q_space, stack_indices)
        # # edges = torch.cdist(q_space, q_space, p=2)  
        # edge_index = torch.nonzero(torch.ones_like(edges_weights), as_tuple=False).T 
        # edge_attr = edges_weights[edge_index[0], edge_index[1]].unsqueeze(0).T
        # rigid_params = self.GNN(com_features.squeeze(1), edge_index, edge_attr)
        rigid_params = self.registration_head(com_features)
        pred_rigid_params = rescale_registration_net_output(rigid_params, self.rigid_parameters_ranges, self.add_to_scale)
        # if epoch%20==0 : plot_feature_similarity(pred_rigid_params, epoch)
        rigid_trans = transformationMatrices(pred_rigid_params[:,:3], pred_rigid_params[:,3:],  self.device)
        # print(f'Rigid Parameters for stack 10 : {pred_rigid_params[10].tolist()}')
        rigid_trans = convert_affine_matrix_from_mm_to_pixels(rigid_trans, self.voxel_size, self.image_size)
        return rigid_trans
    

class RegNet_slice(nn.Module):
    def __init__(self, rigid_config, 
                        GNN_args, 
                        use_GNN=True, 
                        vol_features_size=128, 
                        stack_features_size=128, 
                        layer_to_slice_emb = -1,
                        device='cpu', 
                        k_for_a_matrix = 8,
                        dtype = torch.float32):
        super(RegNet_slice, self).__init__()
        self.VolumeEncoder = generate_Resnet3D(model_depth=34, n_classes=vol_features_size).to(dtype)
        self.SliceEncoder = SliceEncoder(n_features = stack_features_size, d_in = 1).to(dtype)
        # self.GNN = regGNN(in_channels=stack_features_size+vol_features_size, edge_dim=1, hidden_dims=hidden_dims, num_layers=len(hidden_dims))
        if use_GNN:
            self.model = AttnRegGNN(in_channels=stack_features_size+vol_features_size, edge_dim=1, **GNN_args).to(dtype)

        else:
            self.model = SliceFeaturesEncoder(input_dim=stack_features_size+vol_features_size).to(dtype)
        self.rigid_parameters_ranges = rigid_config['ranges']
        self.add_to_scale = rigid_config['add_to_scale']
        self.voxel_size = rigid_config['voxel_size']
        self.image_size = rigid_config['image_size']
        self.device = device
        self.use_GNN = use_GNN
        self.dtype = dtype

    def forward(self, dwi_slices, T1_vol, q_space, stack_indices, output_slice_features = True, epoch=-1):

        # Extract the features vector for each stack of slices and T1 vol
        vol_features = self.VolumeEncoder(T1_vol.unsqueeze(0).unsqueeze(0))
        slices_features = self.SliceEncoder(dwi_slices.unsqueeze(1))
        com_features = torch.concat([slices_features,vol_features.repeat(slices_features.shape[0], 1, 1).squeeze(1)], dim=-1)

        # Extract rigid parameters
        if  self.use_GNN:
            edge_index, edge_attr  = sparse_a_matrix(q_space, self.voxel_size[-1], stack_indices, self.image_size, k=8)
            edge_attr = edge_attr.unsqueeze(1)
            rigid_params, GNN_slice_features  = self.model(com_features.squeeze(1), edge_index, edge_attr.to(self.dtype))
        else:
            rigid_params = self.model(com_features)

        pred_rigid_params = rescale_registration_net_output(rigid_params, self.rigid_parameters_ranges, self.add_to_scale)
        # if epoch%20==0 : plot_feature_similarity(pred_rigid_params, epoch)
        rigid_trans = transformationMatrices(pred_rigid_params[:,:3], pred_rigid_params[:,3:],  self.device)
        rigid_trans = convert_affine_matrix_from_mm_to_pixels(rigid_trans, self.voxel_size, self.image_size)
        if output_slice_features:
            return rigid_trans.to(self.dtype), GNN_slice_features.to(self.dtype)
        return rigid_trans.to(self.dtype)

# class ReconNet(nn.Module):
#     def __init__(self, 
#                  recon_config,
#                  n_vols,
#                  image_size,
#                  device = 'cpu',
#                  dtype = torch.float32):
#         super().__init__()
#         self.INR_encoding = tcnn.Encoding(n_input_dims = 3, encoding_config = recon_config["encoding"]).to(device=device, dtype=dtype)
#         self.INR_networks = torch.nn.ModuleList([
#             tcnn.Network(self.INR_encoding.n_output_dims, 1, recon_config["network"]).to(device=device, dtype=dtype)
#             for _ in range(n_vols)
#             ])

#         self.image_size = image_size
#         self.device = device
#         self.n_vols = n_vols


#     def forward(self, pred_rigid_trans, dwi_stacks, stacks_indices):
#         # inv_trans_per_stack = get_inverse_transformation(pred_rigid_trans)
#         grids = torch.nn.functional.affine_grid(pred_rigid_trans[:,:-1,:], (len(dwi_stacks),1, self.image_size[0],  self.image_size[1],  self.image_size[2]), align_corners=True)
#         grids = order_grids_by_volumes(grids, stacks_indices, device=self.device)
#         recon_volumes = []
#         for i in range(grids.shape[0]):
#             grid = grids[i,...].view(-1,3) #grids[i][...,stacks_indices[i//7],:].view(-1,3)
#             encoded_coord = self.INR_encoding(grid)
#             # print(i%7)
#             recon_vol = self.INR_networks[i%7](encoded_coord)
#             # recon_vol = self.INR_network(encoded_coord)
#             # psf_recon = self.apply_psf_to_recon(recon_vol.to(torch.float32), grid[...,:3])
#             # if i%7==3:
#             recon_volumes.append(recon_vol.to(torch.float32))
#         return recon_volumes
    
#     @torch.no_grad()
#     def get_recons(self,  dwi_stacks ,dwi_slices, T1_vol, q_space_slices, stacks_indices):
#             self.eval()
#             dwi_stacks = [stack.to(self.device) for stack in dwi_stacks]
#             dwi_slices = dwi_slices.to(self.device)
#             T1_image = T1_vol.to(self.device)
#             slices_bvecs = q_space_slices.to(self.device)

                    
#             trans = torch.stack([torch.eye(4)]*self.n_vols)
#             grids = torch.nn.functional.affine_grid(trans[:,:-1,:], (self.n_vols,1,T1_image.shape[0], T1_image.shape[1], T1_image.shape[2]), align_corners=True).to(self.device)
    
         
#             recon_volumes = []
#             for i in range(grids.shape[0]):
#                 grid = grids[i].view(-1,3)#grids[i,...].view(-1,3)
#                 encoded_coord = self.INR_encoding(grid)
#                 recon_vol = self.INR_networks[i](encoded_coord)

#                 recon_volumes.append(recon_vol.to(torch.float32).view(T1_image.shape))
#             return recon_volumes


class ReconNet(nn.Module):
    def __init__(self, recon_config, n_vols, image_size, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_vols = n_vols
        self.image_size = image_size  # (D, H, W)

        # 1) Encoding first, so we know its output dim
        self.INR_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=recon_config["encoding"]
        ).to(device=device, dtype=dtype)

        feat_dim = self.INR_encoding.n_output_dims  # <-- correct input dim

        # 2) One MLP per volume: R^F -> R^1
        self.INR_networks = nn.ModuleList([
            tcnn.Network(
                n_input_dims=feat_dim,
                n_output_dims=1,
                network_config=recon_config["network"]
            ).to(device=device, dtype=dtype)
            for _ in range(n_vols)
        ])

    @staticmethod
    def _to01(coords_minus1_1):
        # If you trained in [-1,1], you can NO-OP here. If you trained in [0,1], convert.
        return (coords_minus1_1 + 1.0) * 0.5

    def forward(self, pred_rigid_trans, dwi_stacks, stacks_indices):
        # Build grid in [-1,1] using the SAME align_corners & spatial size as in get_recons
        N = len(dwi_stacks)  # number of stacks predicted this batch
        grid = torch.nn.functional.affine_grid(
            pred_rigid_trans[:, :-1, :],  # [N,3,4]
            (N, 1, self.image_size[0], self.image_size[1], self.image_size[2]),
            align_corners=True
        )  # [N, D, H, W, 3] in [-1,1]

        # Ensure same device/dtype
        grid = grid.to(self.device, dtype=torch.float32)

        # Reorder stacks -> volumes (make sure this matches get_recons)
        grids = order_grids_by_volumes(grid, stacks_indices, device=self.device)  # [n_vols, D, H, W, 3]

        outs = []
        for vol_idx in range(grids.shape[0]):
            g = grids[vol_idx].reshape(-1, 3)
            # If your encoding expects [0,1]^3, uncomment:
            # g = self._to01(g)
            z = self.INR_encoding(g)                # [V, F]
            y = self.INR_networks[vol_idx](z)       # [V, 1]
            outs.append(y.to(torch.float32))
        return outs

    @torch.no_grad()
    def get_recons(self, dwi_stacks, dwi_slices, T1_vol, q_space_slices, stacks_indices):
        self.eval()  # important

        # Create identity transforms for each volume
        eye44 = torch.eye(4, device=self.device, dtype=torch.float32)
        trans = torch.stack([eye44] * self.n_vols, dim=0)  # [n_vols, 4, 4]
        theta = trans[:, :-1, :]                            # [n_vols, 3, 4]

        # Use the SAME grid size & align_corners as in forward
        D, H, W = T1_vol.shape  # or self.image_size, but be consistent across train/test
        grid = torch.nn.functional.affine_grid(
            theta, (self.n_vols, 1, D, H, W), align_corners=True
        ).to(self.device, dtype=torch.float32)              # [-1,1]

        recons = []
        for vol_idx in range(self.n_vols):
            g = grid[vol_idx].reshape(-1, 3)
            # If trained in [0,1]:
            # g = self._to01(g)
            z = self.INR_encoding(g)
            y = self.INR_networks[vol_idx](z)              # [D*H*W, 1]
            recons.append(y.view(D, H, W).to(torch.float32))
        return recons





class SVRNet(nn.Module):
    def __init__(self, 
                 recon_config, 
                 rigid_config, 
                 vol_features_size=128, 
                 stack_features_size=128, 
                 n_vols = 7,
                 slice_emb_for_inr_size = 16,
                 use_GNN = True, 
                 GNN_args = None, 
                 single_vol_per_epoch_opt = True,
                 recon_n_vols = True,
                 add_qspace2inr = True,
                 layer_to_slice_emb = 3,
                 device='cpu',
                 dtype = torch.float32):
        
        
        super(SVRNet, self).__init__()
        self.reg_model = RegNet_slice(rigid_config = rigid_config,
                                            GNN_args = GNN_args,
                                            vol_features_size = vol_features_size, 
                                            stack_features_size = stack_features_size, 
                                            layer_to_slice_emb = layer_to_slice_emb,
                                            use_GNN = use_GNN,
                                            device=device).to(dtype)
        # enc_out = recon_config["encoding"]["nested"][0]["n_levels"]+recon_config["encoding"]["nested"][1]['n_input_dims']#['n_input_dims']#["n_frequencies"]*6
        # inr_input_size = slice_emb_for_inr_size + enc_out
        # self.INR_encoding = tcnn.Encoding(n_input_dims = 3+recon_config["encoding"]["nested"][1]['n_input_dims'], encoding_config = recon_config["encoding"]).to(dtype)
        self.INR_networks = torch.nn.ModuleList([
            tcnn.Network(recon_config["encoding"]["n_levels"], 1, recon_config["network"]).to(device=device, dtype=dtype)
            for _ in range(n_vols)
            ])
        # self.INR_network = tcnn.Network(recon_config["encoding"]["n_levels"], 1, recon_config["network"]).to(dtype)
        self.INR_encoding = tcnn.Encoding(n_input_dims = 3, encoding_config = recon_config["encoding"]).to(dtype)
        # self.INR_network = tcnn.Network(16, 1, recon_config["network"]).to(dtype)
       

        # self.INR = tcnn.NetworkWithInputEncoding(stack_emb_for_inr_size+3, 1, recon_config["encoding"], recon_config["network"] )
        if slice_emb_for_inr_size !=0:
            self.slice_encoder = SliceFeaturesEncoder(input_dim=128, output_dim=slice_emb_for_inr_size)
        self.rigid_parameters_ranges = rigid_config['ranges']
        self.add_to_scale = rigid_config['add_to_scale']
        self.voxel_size = rigid_config['voxel_size']
        self.image_size = rigid_config['image_size']
        self.n_vols = n_vols
        self.device = device
        self.slice_emb_for_inr_size = slice_emb_for_inr_size
        self.add_qspace2inr = False#add_qspace2inr
        self.single_vol_per_epoch_opt = single_vol_per_epoch_opt
    
        self.recon_n_vols = recon_n_vols

    def forward(self, dwi_stacks ,dwi_slices, T1_vol, q_space_slices, stacks_indices,  iter= -1):

        rigid_trans_per_slice, slices_features = self.reg_model(dwi_slices, T1_vol, q_space_slices, stacks_indices)

        rigid_trans_per_stack, reg_losses, var_losses = apply_rigid_trans_slice2stacks(rigid_trans_per_slice, dwi_stacks, stacks_indices, self.n_vols, T1_vol = T1_vol, calc_reg_MI = True, iter = iter) 
        inv_trans_per_stack = get_inverse_transformation(rigid_trans_per_stack)

        if self.slice_emb_for_inr_size !=0:
            # slices_emb = self.slice_encoder(slices_features)
            slices_emb = slices_features.view(self.n_vols, self.image_size[-1], -1)[:,None, None,...]
            slices_emb = slices_emb.expand(-1, self.image_size[0],  self.image_size[1],  -1, -1)#.detach()
            # stacks_emb = get_stack_features_from_slice_features(slices_features, stacks_indices, len(dwi_stacks)//len(stacks_indices), len(dwi_stacks))
            # stacks_emb = self.stacks_encoding(torch.stack(stacks_emb))
            # stacks_emb = stacks_emb[:, None, None, None, :]  
            # stacks_emb = stacks_emb.expand(-1, self.image_size[0],  self.image_size[1],  self.image_size[2], -1).detach()
        
        if self.add_qspace2inr and self.slice_emb_for_inr_size==0:
            q_space_stacks = get_stack_features_from_slice_features(q_space_slices, stacks_indices, len(dwi_stacks)//len(stacks_indices), len(dwi_stacks))
            q_space_stacks = torch.stack(q_space_stacks)
            q_space_stacks = q_space_stacks[:, None, None, None, :]  
            q_space_stacks = q_space_stacks.expand(-1, self.image_size[0],  self.image_size[1],  self.image_size[2], -1)
        # rigid_trans = transformationMatrices(pred_rigid_params[:,:3], pred_rigid_params[:,3:],  self.device)
        # rigid_trans = convert_affine_matrix_from_mm_to_pixels(rigid_trans, self.voxel_size, self.image_size)
        grids = torch.nn.functional.affine_grid(inv_trans_per_stack[:,:-1,:], (len(dwi_stacks),1, self.image_size[0],  self.image_size[1],  self.image_size[2]), align_corners=True)
        # transformed_grid = transform_grid(grids, rigidinv_trans_per_stack_trans)
        
        if self.recon_n_vols:
            grids = order_grids_by_volumes(grids, stacks_indices, device=self.device)

        if self.single_vol_per_epoch_opt:
            # choose random vol to optimize 
            self.vol_to_opt = torch.randint(0, self.n_vols, (1,)).item()
            if self.slice_emb_for_inr_size !=0:
                grid = torch.cat([grids[self.vol_to_opt,...].view(-1,3), slices_emb[self.vol_to_opt,...].reshape(-1, self.slice_emb_for_inr_size)], -1)  
            elif self.add_qspace2inr:
                grid = torch.cat([grids[self.vol_to_opt,...].view(-1,3), q_space_stacks[self.vol_to_opt,...].view(-1, 3)], -1)  
            else:
                grid = grids[self.vol_to_opt,...].view(-1,3)
            encoded_coord = self.INR_encoding(grid)
            recon_vol = self.INR_networks[self.vol_to_opt](encoded_coord).to(torch.float32)
            return [recon_vol.view(self.image_size[0],  self.image_size[1],  self.image_size[2]), reg_losses, var_losses]
        else:
            recon_volumes = []
            for i in range(grids.shape[0]):
                if self.slice_emb_for_inr_size !=0:
                    grid = torch.cat([grids[i,...].view(-1,3), slices_emb[i,...].reshape(-1, self.slice_emb_for_inr_size)], -1)  
                elif self.add_qspace2inr:
                    grid = torch.cat([grids[i,...].view(-1,3), q_space_stacks[i,...].view(-1, 3)], -1)  
                else:
                    grid = grids[i][...,stacks_indices[i//7],:].view(-1,3)#grids[i,...].view(-1,3)
                encoded_coord = self.INR_encoding(grid)
                # print(i%7)
                recon_vol = self.INR_networks[i%7](encoded_coord)
                # recon_vol = self.INR_network(encoded_coord)
                # psf_recon = self.apply_psf_to_recon(recon_vol.to(torch.float32), grid[...,:3])
                # if i%7==3:
                recon_volumes.append(recon_vol.to(torch.float32))
                # else:
                #     recon_volumes.append(recon_vol.to(torch.float32).detach())
            # return recon_volumes.view(-1, self.image_size[0],  self.image_size[1],  self.image_size[2])
            return [recon_volumes, reg_losses, var_losses]

            # return [torch.stack(recon_volumes).view(-1,self.image_size[0],  self.image_size[1],  self.image_size[2]), reg_losses, var_losses]

    def apply_psf_to_recon(self, recon_vol):
        psf = get_PSF(r_max=None,
                      res_ratio=self.voxel_size,
                      psf_type="gaussian",
                      device=self.device)
        
        psf_recon = apply_psf(recon_vol.view(self.image_size.tolist()), psf, device=self.device)
        # psf_recon = apply_psf_at_points(recon_vol.view(self.image_size.tolist()), psf, grid, device=self.device)
        return psf_recon
        # outs = []
        # for i in range(0, grid.shape[0], chunk):
        #     outs.append(apply_psf_at_points(recon_vol.view(self.image_size.tolist()), psf, grid[i:i+chunk], device=self.device))
        #     torch.cuda.empty_cache()
        # return torch.cat(outs)
        

    def get_recons(self,  dwi_stacks ,dwi_slices, T1_vol, q_space_slices, stacks_indices):
            
            rigid_trans_per_slice, slices_features = self.reg_model(dwi_slices, T1_vol, q_space_slices, stacks_indices)
            dwi_stacks = [stack.to(self.device) for stack in dwi_stacks]
            dwi_slices = dwi_slices.to(self.device)
            T1_image = T1_vol.to(self.device)
            slices_bvecs = q_space_slices.to(self.device)

                    
            if self.add_qspace2inr and self.slice_emb_for_inr_size==0:
                q_space_stacks = get_stack_features_from_slice_features(q_space_slices, stacks_indices, len(dwi_stacks)//len(stacks_indices), len(dwi_stacks))
                q_space_stacks = torch.stack(q_space_stacks)
                q_space_stacks = q_space_stacks[:, None, None, None, :]  
                q_space_stacks = q_space_stacks.expand(-1, self.image_size[0],  self.image_size[1],  self.image_size[2], -1)


            if self.slice_emb_for_inr_size !=0:
                slices_emb = self.slice_encoder(slices_features)
                slices_emb = slices_emb.view(self.n_vols, self.image_size[-1], -1)[:,None, None,...]
                slices_emb = slices_emb.expand(-1, self.image_size[0],  self.image_size[1],  -1, -1)#.detach()
                # stacks_emb = get_stack_features_from_slice_features(slices_features, stacks_indices, len(dwi_stacks)//len(stacks_indices), len(dwi_stacks))
                # stacks_emb = self.stacks_encoding(torch.stack(stacks_emb))
                # stacks_emb = stacks_emb[:, None, None, None, :]  
                # stacks_emb = stacks_emb.expand(-1, self.image_size[0],  self.image_size[1],  self.image_size[2], -1)

            trans = torch.stack([torch.eye(4)]*self.n_vols)
            grids = torch.nn.functional.affine_grid(trans[:,:-1,:], (self.n_vols,1,T1_image.shape[0], T1_image.shape[1], T1_image.shape[2]), align_corners=True).to(self.device)
    
            # trans = torch.stack([torch.eye(4)]*len(dwi_stacks))
            # grids = torch.nn.functional.affine_grid(trans[:,:-1,:], (len(dwi_stacks),1,T1_image.shape[0], T1_image.shape[1], T1_image.shape[2]), align_corners=True).to(self.device)
            if self.recon_n_vols:
                grids = order_grids_by_volumes(grids, stacks_indices, device=self.device)
            else:
                grids = grids
            recon_volumes = []
            for i in range(grids.shape[0]):
                if self.slice_emb_for_inr_size !=0:
                    grid = torch.cat([grids[i,...].view(-1,3), slices_emb[i,...].view(-1, self.stack_emb_for_inr_size)], -1)  
                elif self.add_qspace2inr:
                    grid = torch.cat([grids[i,...].view(-1,3), q_space_stacks[i,...].view(-1, 3)], -1)  
                else:
                    grid = grids[i].view(-1,3)#grids[i,...].view(-1,3)
                encoded_coord = self.INR_encoding(grid)
                recon_vol = self.INR_networks[i%7](encoded_coord)
                # recon_vol = self.INR_network(encoded_coord)

                # psf_recon = self.apply_psf_to_recon(recon_vol.to(torch.float32))
                # recon_volumes.append(psf_recon)

                recon_volumes.append(recon_vol.to(torch.float32).view(T1_image.shape))
            return recon_volumes