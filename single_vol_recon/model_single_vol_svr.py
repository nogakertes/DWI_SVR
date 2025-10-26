import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, JumpingKnowledge, GraphNorm
import torch.nn.functional as F
import sys
sys.path.append('.')
from encoders import generate_Resnet3D,ResNet2D, Encoder2D
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
import tinycudann as tcnn
from contextlib import nullcontext
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
    

class registration_head(nn.Module):
    def __init__(self, input_dim=128, output_dim=6):
        super(registration_head, self).__init__()
        
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

    

import torch
import torch.nn as nn
import tinycudann as tcnn
from contextlib import nullcontext

class ReconNet(nn.Module):
    def __init__(self, recon_config, image_size,
                 slice_axis='W', psf_fwhm01=1.0/128.0, psf_nsamples=1,
                 device='cuda', dtype=torch.float32,
                 amp: bool = True,                  # <-- enable mixed precision
                 query_chunk_points: int = 4096   # <-- INR queries per chunk (~128k)
                 ):
        super().__init__()
        self.device = device
        self.dtype  = dtype
        self.image_size = tuple(image_size)  # (D,H,W)
        assert slice_axis in {'W','H','D'}
        self.slice_axis = slice_axis
        self.amp = amp
        self.query_chunk_points = int(query_chunk_points)

        # tcnn expects coords in [0,1]
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=recon_config["encoding"]
        ).to(device=device, dtype=dtype)
        feat_dim = self.encoding.n_output_dims

        self.mlp = tcnn.Network(
            n_input_dims=feat_dim,
            n_output_dims=1,
            network_config=recon_config["network"]
        ).to(device=device, dtype=dtype)

        # in-plane (U,V) grid for the slice plane (in [0,1])
        D, H, W = self.image_size
        if self.slice_axis == 'W':   # planes are H×W
            self.Hp, self.Wp = H, W
        elif self.slice_axis == 'H': # planes are D×W
            self.Hp, self.Wp = D, W
        else:                        # 'D' planes are H×W
            self.Hp, self.Wp = H, W

        u = torch.linspace(0.0, 1.0, steps=self.Wp, device=device, dtype=torch.float32)
        v = torch.linspace(0.0, 1.0, steps=self.Hp, device=device, dtype=torch.float32)
        V, U = torch.meshgrid(v, u, indexing='ij')       # [Hp,Wp]
        base_plane = torch.stack([U, V], dim=-1)         # [Hp,Wp,2]
        self.register_buffer("base_plane_uv01", base_plane, persistent=False)

        # PSF samples along the slice normal (Gaussian)
        self.psf_nsamples = int(psf_nsamples)
        sigma = (psf_fwhm01 / 2.355)
        t = torch.linspace(-2.5*sigma, 2.5*sigma, steps=self.psf_nsamples,
                           device=device, dtype=torch.float32)
        w = torch.exp(-0.5 * (t / (sigma + 1e-8)).pow(2))
        w = (w / (w.sum() + 1e-8)).contiguous()
        self.register_buffer("psf_offsets01", t, persistent=False)   # [S]
        self.register_buffer("psf_weights",   w, persistent=False)   # [S]

    @staticmethod
    def _pad3x4_to4x4(theta3x4: torch.Tensor) -> torch.Tensor:
        N = theta3x4.shape[0]
        eye = torch.eye(4, device=theta3x4.device, dtype=theta3x4.dtype).unsqueeze(0).repeat(N,1,1)
        eye[:, :3, :] = theta3x4
        return eye

    def _slice_plane_to_canonical(self, Theta01_44_inv: torch.Tensor,
                                  z01: torch.Tensor) -> torch.Tensor:
        """
        Back-map in-plane pixels at slice position z01 (in [0,1]) to canonical coords.
        Returns [HW, 3] in [0,1].
        """
        UV = self.base_plane_uv01  # [Hp,Wp,2]
        Hp, Wp = UV.shape[:2]
        ones = torch.ones(Hp, Wp, 1, device=UV.device, dtype=UV.dtype)

        if self.slice_axis == 'W':
            xyz = torch.cat([UV, z01.expand(Hp, Wp, 1)], dim=-1)
        elif self.slice_axis == 'H':
            # (x=U, y=z01, z=V)
            xyz = torch.cat([UV[..., :1], z01.expand(Hp, Wp, 1), UV[..., 1:]], dim=-1)
        else:  # 'D'
            # (x=V, y=U, z=z01)
            x = UV[..., 1:2]; y = UV[..., 0:1]
            xyz = torch.cat([x, y, z01.expand(Hp, Wp, 1)], dim=-1)

        Xh_slice = torch.cat([xyz, ones], dim=-1).view(-1, 4)          # [HW,4]
        X_canon  = (Xh_slice @ Theta01_44_inv.T)[:, :3]                # [HW,3]
        return X_canon

    def _query_inr_chunked(self, coords01_flat: torch.Tensor) -> torch.Tensor:
        """
        Memory-safe INR queries on [V,3] coords → [V].
        Uses chunking and optional AMP to reduce CUDA memory footprint.
        """
        V = coords01_flat.shape[0]
        out = torch.empty(V, device=self.device, dtype=torch.float32)   # keep output in fp32
        autocast_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                        if (self.amp and self.device.startswith('cuda')) else nullcontext())
        s = 0
        while s < V:
            e = min(s + self.query_chunk_points, V)
            with autocast_ctx:
                z = self.encoding(coords01_flat[s:e].contiguous())
                y = self.mlp(z).squeeze(-1)            # dtype=fp16 if amp, else fp32
            out[s:e] = y.float()
            s = e
        return out

    def forward(self, theta01_batch, dwi_stacks, stacks_indices):
        """
        theta01_batch : [N,3,4] or [N,4,4], canonical[0,1] → slice[0,1]
        dwi_stacks    : list of N tensors [D,H,W] (targets)
        stacks_indices: list of N (slice indices along slice_axis)

        Returns: list of N simulated stacks, each [Hp, Wp, K_i]
        """
        N = len(dwi_stacks)

        # normalize theta to [N,4,4]
        if theta01_batch.shape[-2:] == (3,4):
            Theta44 = self._pad3x4_to4x4(theta01_batch.to(self.device, self.dtype))
        elif theta01_batch.shape[-2:] == (4,4):
            Theta44 = theta01_batch.to(self.device, self.dtype)
        else:
            raise ValueError("theta must be [N,3,4] or [N,4,4] in [0,1] space (canonical→slice)")

        Theta_inv = torch.linalg.inv(Theta44)  # [N,4,4]

        Hp, Wp = self.Hp, self.Wp
        sims = []

        for i in range(N):
            stack = dwi_stacks[i].to(self.device, torch.float32)  # [D,H,W] (only for sizes)
            idxs = stacks_indices[i].tolist() if torch.is_tensor(stacks_indices[i]) else list(stacks_indices[i])
            K = len(idxs)

            D, H, W = stack.shape
            if self.slice_axis == 'W':
                size_axis = W
            elif self.slice_axis == 'H':
                size_axis = H
            else:
                size_axis = D

            # pre-allocate output (fp32)
            sim = torch.empty(Hp, Wp, K, device=self.device, dtype=torch.float32)

            # canonical normal from slice +z direction
            n_slice = torch.tensor([0., 0., 1., 0.], device=self.device, dtype=torch.float32)
            n_canon = (Theta_inv[i] @ n_slice)[:3]
            n_canon = n_canon / (n_canon.norm() + 1e-8)

            for j, k in enumerate(idxs):
                # slice position in [0,1]
                z01 = torch.tensor([k / max(1, (size_axis - 1))],
                                   device=self.device, dtype=torch.float32).expand(1)
                # base canonical points for this plane
                X0 = self._slice_plane_to_canonical(Theta_inv[i], z01)  # [HW,3]

                # integrate along normal with PSF, in chunks
                accum = torch.zeros(Hp*Wp, device=self.device, dtype=torch.float32)
                for t_off, w in zip(self.psf_offsets01, self.psf_weights):
                    Xk = (X0 + t_off * n_canon).clamp(0.0, 1.0)          # [HW,3]
                    Ik = self._query_inr_chunked(Xk)                      # [HW] fp32
                    accum.add_(w * Ik)                                    # in-place accumulate

                sim[..., j] = accum.view(Hp, Wp)

            sims.append(sim)

        return sims

    @torch.no_grad()
    def get_recons(self, chunk_vox: int = 256**2):
        """
        Chunked canonical evaluation on identity grid to produce [D,H,W].
        """
        self.eval()
        D, H, W = self.image_size
        z = torch.linspace(0.0, 1.0, D, device=self.device)
        y = torch.linspace(0.0, 1.0, H, device=self.device)
        x = torch.linspace(0.0, 1.0, W, device=self.device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)  # [DHW,3]

        out = torch.empty(D*H*W, device=self.device, dtype=torch.float32)
        autocast_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                        if (self.amp and self.device.startswith('cuda')) else nullcontext())

        s = 0
        while s < coords.shape[0]:
            e = min(s + chunk_vox, coords.shape[0])
            with autocast_ctx:
                zf = self.encoding(coords[s:e].contiguous())
                yf = self.mlp(zf).squeeze(-1)
            out[s:e] = yf.float()
            s = e

        return out.view(D, H, W)

        
# class ReconNet(nn.Module):
#     """
#     INR reconstructor that expects theta in [0,1] space (fixed_01 -> moving_01).

#     Args:
#         recon_config: tinycudann encoding + network configs
#         image_size  : (D, H, W) integer tuple
#         device      : torch device
#         dtype       : torch dtype
#     """
#     def __init__(self, recon_config, image_size, device='cpu', dtype=torch.float32):
#         super().__init__()
#         self.device = device
#         self.dtype = dtype
#         self.image_size = tuple(image_size)  # (D, H, W)

#         # Encoding expects coords in [0,1]
#         self.INR_encoding = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config=recon_config["encoding"]
#         ).to(device=device, dtype=dtype)

#         feat_dim = self.INR_encoding.n_output_dims

#         self.INR_network = tcnn.Network(
#             n_input_dims=feat_dim,
#             n_output_dims=1,
#             network_config=recon_config["network"]
#         ).to(device=device, dtype=dtype)

#         # Precompute base fixed-grid in [0,1], shaped [1,D,H,W,3]
#         self.register_buffer(
#             "_base_grid_01",
#             self._make_base_grid_01(*self.image_size),
#             persistent=False
#         )

#     # ---------- grids & transforms in [0,1] space ----------

#     @staticmethod
#     def _make_base_grid_01(D, H, W, device=None, dtype=torch.float32):
#         """
#         Create a fixed grid in [0,1]^3 with ij indexing.
#         Returns [1, D, H, W, 3] with channels = (x,y,z) each in [0,1].
#         """
#         z = torch.linspace(0.0, 1.0, steps=D, device=device, dtype=dtype)
#         y = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=dtype)
#         x = torch.linspace(0.0, 1.0, steps=W, device=device, dtype=dtype)
#         zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")  # [D,H,W]
#         grid = torch.stack([xx, yy, zz], dim=-1)             # [D,H,W,3]
#         return grid.unsqueeze(0)                              # [1,D,H,W,3]

#     def _apply_theta01_to_grid01(self, theta01_batch: torch.Tensor, base_grid_01: torch.Tensor):
#         """
#         theta01_batch : [N,3,4] or [N,4,4], mapping fixed_01 -> moving_01
#         base_grid_01  : [1,D,H,W,3], in fixed_01
#         returns       : [N,D,H,W,3], in moving_01
#         """
#         if theta01_batch.shape[-2:] == (4,4):
#             theta01 = theta01_batch[:, :3, :]     # reduce to [N,3,4]
#         elif theta01_batch.shape[-2:] == (3,4):
#             theta01 = theta01_batch
#         else:
#             raise ValueError("theta must be [N,3,4] or [N,4,4] in [0,1] space")

#         N = theta01.shape[0]
#         D, H, W = base_grid_01.shape[1:4]

#         # Flatten base grid and add homogeneous 1s
#         G = base_grid_01.view(1, -1, 3)                                   # [1,V,3]
#         ones = torch.ones((1, G.shape[1], 1), device=G.device, dtype=G.dtype)
#         G_h = torch.cat([G, ones], dim=-1).expand(N, -1, -1)              # [N,V,4]

#         # Right-multiply by theta^T to apply [x,y,z,1] @ theta -> [x',y',z']
#         Theta_T = theta01.transpose(1, 2).contiguous()                    # [N,4,3]
#         moved = torch.bmm(G_h, Theta_T)                                   # [N,V,3]
#         return moved.view(N, D, H, W, 3)

#     # ---------- model API ----------

#     def forward(self, theta01_batch, dwi_stacks, stacks_indices):
#         """
#         theta01_batch : [N,3,4] or [N,4,4], mapping fixed_01 -> moving_01
#         dwi_stacks    : list of N stacks (only used for N)
#         stacks_indices: per-sample indices along W axis to evaluate (list or 1D tensors)
#         returns       : list of N tensors, each [V_sel] (flattened selected voxels)
#         """
#         N = len(dwi_stacks)
#         base = self._base_grid_01.to(self.device, dtype=torch.float32)               # [1,D,H,W,3]
#         grids01 = self._apply_theta01_to_grid01(theta01_batch.to(self.device, self.dtype), base)  # [N,D,H,W,3]

#         outs = []
#         for i in range(N):
#             # pick only the requested slices along W, then flatten to [V_sel, 3]
#             sel = stacks_indices[i].tolist() if torch.is_tensor(stacks_indices[i]) else stacks_indices[i]
#             g01 = grids01[i, :, :, sel, :].contiguous().view(-1, 3)                 # [V_sel,3] in [0,1]
#             z = self.INR_encoding(g01)                                              # [V_sel, F]
#             y = self.INR_network(z).squeeze(-1).contiguous()                        # [V_sel]
#             outs.append(y.to(torch.float32))
#         return outs

#     @torch.no_grad()
#     def get_recons(self):
#         """
#         Produce a full reconstruction on the canonical fixed grid (identity warp).
#         """
#         self.eval()
#         D, H, W = self.image_size
#         # Identity in [0,1] space: [I|0]
#         theta_I = torch.eye(4, device=self.device, dtype=torch.float32)[:3, :].unsqueeze(0)  # [1,3,4]
#         base = self._base_grid_01.to(self.device, dtype=torch.float32)                       # [1,D,H,W,3]
#         grid01 = self._apply_theta01_to_grid01(theta_I, base)[0]                             # [D,H,W,3]

#         z = self.INR_encoding(grid01.view(-1, 3))
#         y = self.INR_network(z).squeeze(-1).contiguous()                                     # [D*H*W]
#         return y.view(D, H, W).to(torch.float32)

# class ReconNet(nn.Module):
#     def __init__(self, recon_config, image_size, device='cpu', dtype=torch.float32):
#         super().__init__()
#         self.device = device
#         self.dtype = dtype
#         self.image_size = image_size  # (D, H, W)

#         # 1) Encoding first, so we know its output dim
#         self.INR_encoding = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config=recon_config["encoding"]
#         ).to(device=device, dtype=dtype)

#         feat_dim = self.INR_encoding.n_output_dims  # <-- correct input dim

#         self.INR_network =tcnn.Network(
#                 n_input_dims=feat_dim,
#                 n_output_dims=1,
#                 network_config=recon_config["network"]
#             ).to(device=device, dtype=dtype)
    

#     @staticmethod
#     def _to01(coords_minus1_1):
#         # Map from [-1,1] (affine_grid) -> [0,1] (HashGrid expects this!)
#         return (coords_minus1_1 + 1.0) * 0.5

#     def forward(self, pred_rigid_trans, dwi_stacks, stacks_indices):
#         # Build grid in [-1,1] using the SAME align_corners & spatial size as in get_recons
#         N = len(dwi_stacks)  # number of stacks predicted this batch
#         # pred_rigid_trans = get_inverse_transformation(pred_rigid_trans)
#         # grids = torch.nn.functional.affine_grid(
#         #     pred_rigid_trans[:, :-1, :],  # [N,3,4]
#         #     (N, 1, self.image_size[0], self.image_size[1], self.image_size[2]),
#         #     align_corners=True)  # [N, D, H, W, 3] in [-1,1]
#         rots, trans = extract_rigid_parameters(pred_rigid_trans)
#         rots = torch.stack([rots[:,0], rots[:,1], rots[:,2]]).T
#         trans = torch.stack([trans[:,0], trans[:,1], trans[:,2]]).T
#         fixed_rigid_params = transformationMatrices(rots, trans)
#         grids = torch.nn.functional.affine_grid(
#             fixed_rigid_params[:, :-1, :],  # [N,3,4]
#             (N, 1, self.image_size[0], self.image_size[1], self.image_size[2]),
#             align_corners=True)
#         # grids = torch.nn.functional.affine_grid(
#         #     pred_rigid_trans[:, :-1, :],  # [N,3,4]
#         #     (N, 1, self.image_size[0], self.image_size[1], self.image_size[2]),
#         #     align_corners=True)
#         # grids = grids.permute(0,3,2,1,4)
#         # _, grids = inr_coords_from_theta_batch(pred_rigid_trans[:,:-1,:], size_dst_xyz=[128,128,82])
#         # grids = get_grids_for_batch(pred_rigid_trans[:,:-1,:], size_dst_xyz=[128,128,82])#.permute(0,2,3,1,4)

#         # Ensure same device/dtype
#         grids = grids.to(self.device, dtype=torch.float32)


#         outs = []
#         for vol_idx in range(grids.shape[0]):
#             # g = grids[vol_idx,stacks_indices[vol_idx].tolist(),...].view(-1, 3)
#             g = grids[vol_idx,:,:,stacks_indices[vol_idx].tolist(),:].contiguous().view(-1, 3)
#             g01 = self._to01(g)                                    # -> [0,1]
#             z = self.INR_encoding(g01)  
#             y = self.INR_network(z).squeeze(-1).contiguous()         # [V, 1]
#             outs.append(y.to(torch.float32))
#         return outs

#     @torch.no_grad()
#     def get_recons(self):
#         self.eval()  # important

#         # Create identity transforms for each volume
#         eye44 = torch.eye(4, device=self.device, dtype=torch.float32)

#         # Use the SAME grid size & align_corners as in forward
#         D, H, W = self.image_size # or self.image_size, but be consistent across train/test
#         # grid = get_grids_for_batch(eye44[:-1,:].unsqueeze(0), size_dst_xyz=[128,128,82])#.permute(0,2,3,1,4)

#         # grid = torch.nn.functional.affine_grid(
#         #    eye44[:-1,:].unsqueeze(0),  # [N,3,4]
#         #     (1, 1, self.image_size[2], self.image_size[1], self.image_size[0]),
#         #     align_corners=True)
#         # grid = grid.permute(0,3,2,1,4)
#         grid = torch.nn.functional.affine_grid(
#             eye44[:-1,:].unsqueeze(0), (1, 1, D, H, W), align_corners=True
#         ).to(self.device, dtype=torch.float32) 
 
#         g01 = self._to01(grid.contiguous().view(-1, 3))
#         z = self.INR_encoding(g01)
#         y = self.INR_network(z).squeeze(-1).contiguous()                  # [D*H*W, 1]
#         return y.view(D,H,W).to(torch.float32)

class ReconNet(nn.Module):
    def __init__(self, recon_config, image_size, device='cpu', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.image_size = image_size  # (D, H, W)

        # 1) Encoding first, so we know its output dim
        self.INR_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=recon_config["encoding"]
        ).to(device=device, dtype=dtype)

        feat_dim = self.INR_encoding.n_output_dims  # <-- correct input dim

        self.INR_network =tcnn.Network(
                n_input_dims=feat_dim,
                n_output_dims=1,
                network_config=recon_config["network"]
            ).to(device=device, dtype=dtype)
    

    @staticmethod
    def _to01(coords_minus1_1):
        # Map from [-1,1] (affine_grid) -> [0,1] (HashGrid expects this!)
        return (coords_minus1_1 + 1.0) * 0.5

    def forward(self, pred_rigid_trans, dwi_stacks, stacks_indices):
        # Build grid in [-1,1] using the SAME align_corners & spatial size as in get_recons
        N = len(dwi_stacks)  # number of stacks predicted this batch
        # pred_rigid_trans = get_inverse_transformation(pred_rigid_trans)
        # grids = torch.nn.functional.affine_grid(
        #     pred_rigid_trans[:, :-1, :],  # [N,3,4]
        #     (N, 1, self.image_size[0], self.image_size[1], self.image_size[2]),
        #     align_corners=True)  # [N, D, H, W, 3] in [-1,1]
        rots, trans = extract_rigid_parameters(pred_rigid_trans)
        rots = torch.stack([rots[:,0], rots[:,1], rots[:,2]]).T
        trans = torch.stack([trans[:,0], trans[:,1], trans[:,2]]).T
        fixed_rigid_params = transformationMatrices(rots, trans)
        grids = torch.nn.functional.affine_grid(
            fixed_rigid_params[:, :-1, :],  # [N,3,4]
            (N, 1, self.image_size[0], self.image_size[1], self.image_size[2]),
            align_corners=True)
        # grids = torch.nn.functional.affine_grid(
        #     pred_rigid_trans[:, :-1, :],  # [N,3,4]
        #     (N, 1, self.image_size[0], self.image_size[1], self.image_size[2]),
        #     align_corners=True)
        # grids = grids.permute(0,3,2,1,4)
        # _, grids = inr_coords_from_theta_batch(pred_rigid_trans[:,:-1,:], size_dst_xyz=[128,128,82])
        # grids = get_grids_for_batch(pred_rigid_trans[:,:-1,:], size_dst_xyz=[128,128,82])#.permute(0,2,3,1,4)

        # Ensure same device/dtype
        grids = grids.to(self.device, dtype=torch.float32)


        outs = []
        for vol_idx in range(grids.shape[0]):
            # g = grids[vol_idx,stacks_indices[vol_idx].tolist(),...].view(-1, 3)
            # g = grids[vol_idx,:,:,stacks_indices[vol_idx].tolist(),:].contiguous().view(-1, 3)
            g = grids[vol_idx,...].contiguous().view(-1, 3)
            g01 = self._to01(g)                                    # -> [0,1]
            z = self.INR_encoding(g01)  
            y = self.INR_network(z).squeeze(-1).contiguous()         # [V, 1]
            outs.append(y.to(torch.float32))
        return outs

    @torch.no_grad()
    def get_recons(self):
        self.eval()  # important

        # Create identity transforms for each volume
        eye44 = torch.eye(4, device=self.device, dtype=torch.float32)

        # Use the SAME grid size & align_corners as in forward
        D, H, W = self.image_size # or self.image_size, but be consistent across train/test
        # grid = get_grids_for_batch(eye44[:-1,:].unsqueeze(0), size_dst_xyz=[128,128,82])#.permute(0,2,3,1,4)

        # grid = torch.nn.functional.affine_grid(
        #    eye44[:-1,:].unsqueeze(0),  # [N,3,4]
        #     (1, 1, self.image_size[2], self.image_size[1], self.image_size[0]),
        #     align_corners=True)
        # grid = grid.permute(0,3,2,1,4)
        grid = torch.nn.functional.affine_grid(
            eye44[:-1,:].unsqueeze(0), (1, 1, D, H, W), align_corners=True
        ).to(self.device, dtype=torch.float32) 
 
        g01 = self._to01(grid.contiguous().view(-1, 3))
        z = self.INR_encoding(g01)
        y = self.INR_network(z).squeeze(-1).contiguous()                  # [D*H*W, 1]
        return y.view(D,H,W).to(torch.float32)




# class ReconNet(nn.Module):
#     def __init__(self, recon_config, image_size, device='cpu', dtype=torch.float32):
#         super().__init__()
#         self.device = device
#         self.dtype = dtype
#         self.image_size = image_size  # (D, H, W)

#         # 1) Encoding first, so we know its output dim
#         self.INR_encoding = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config=recon_config["encoding"]
#         ).to(device=device, dtype=dtype)

#         feat_dim = self.INR_encoding.n_output_dims  # <-- correct input dim

#         self.INR_network =tcnn.Network(
#                 n_input_dims=feat_dim,
#                 n_output_dims=1,
#                 network_config=recon_config["network"]
#             ).to(device=device, dtype=dtype)
        
#         # Precompute base fixed-grid in [0,1], shaped [1,D,H,W,3]
#         self.register_buffer(
#             "_base_grid_01",
#             self._make_base_grid_01(*self.image_size),
#             persistent=False
#         )
    

#     @staticmethod
#     def _to01(coords_minus1_1):
#         # Map from [-1,1] (affine_grid) -> [0,1] (HashGrid expects this!)
#         return (coords_minus1_1 + 1.0) * 0.5

#     def forward(self, pred_rigid_trans, dwi_stacks, stacks_indices):
#         G = self.base_grid_01.view(1, -1, 3)
#         z = self.INR_encoding(G)  
#         y = self.INR_network(z).squeeze(-1).contiguous()         # [V, 1]
#         return y

#     @torch.no_grad()
#     def get_recons(self):
#         self.eval()  # important

#         # Create identity transforms for each volume
#         eye44 = torch.eye(4, device=self.device, dtype=torch.float32)

#         # Use the SAME grid size & align_corners as in forward
#         D, H, W = self.image_size # or self.image_size, but be consistent across train/test
#         # grid = get_grids_for_batch(eye44[:-1,:].unsqueeze(0), size_dst_xyz=[128,128,82])#.permute(0,2,3,1,4)

#         # grid = torch.nn.functional.affine_grid(
#         #    eye44[:-1,:].unsqueeze(0),  # [N,3,4]
#         #     (1, 1, self.image_size[2], self.image_size[1], self.image_size[0]),
#         #     align_corners=True)
#         # grid = grid.permute(0,3,2,1,4)
#         grid = torch.nn.functional.affine_grid(
#             eye44[:-1,:].unsqueeze(0), (1, 1, D, H, W), align_corners=True
#         ).to(self.device, dtype=torch.float32) 
 
#         g01 = self._to01(grid.contiguous().view(-1, 3))
#         z = self.INR_encoding(g01)
#         y = self.INR_network(z).squeeze(-1).contiguous()                  # [D*H*W, 1]
#         return y.view(D,H,W).to(torch.float32)

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn

class ReconNet_canonial_grid(nn.Module):
    """
    INR predicts the canonical volume (coords in [0,1]).
    For the loss, we warp that volume into each stack using theta in [-1,1]
    with torch.affine_grid + grid_sample (no PSF).

    Assumptions
    -----------
    - theta_m11_batch is [N,3,4] or [N,4,4] in [-1,1] space.
    - Default: theta_m11_batch maps canonical → stack (set theta_is_canon2stack=False
      if you pass stack → canonical instead).
    - align_corners=True everywhere you created/convered theta.
    """

    def __init__(self, recon_config, image_size, device='cuda', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype  = dtype
        self.image_size = tuple(image_size)  # (D,H,W)

        # tcnn: coords ∈ [0,1]^3
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=recon_config["encoding"]
        ).to(device=device, dtype=dtype)

        feat_dim = self.encoding.n_output_dims

        self.mlp = tcnn.Network(
            n_input_dims=feat_dim,
            n_output_dims=1,
            network_config=recon_config["network"]
        ).to(device=device, dtype=dtype)

        # Precomputed canonical grid in [0,1]: [1,D,H,W,3]
        self.register_buffer(
            "_base_grid_01",
            self._make_base_grid_01(*self.image_size, device=device),
            persistent=False
        )

    # ---------- helpers ----------

    @staticmethod
    def _make_base_grid_01(D, H, W, device=None, dtype=torch.float32):
        """
        Create a canonical base grid in [0,1]^3 using affine_grid.
        Returns tensor of shape [1,D,H,W,3].
        """
        # Identity affine: output[-1,1] -> input[-1,1]
        theta = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0)  # [1,3,4]

        # Build affine_grid in [-1,1] coordinates
        grid_m11 = F.affine_grid(theta, size=(1,1,D,H,W), align_corners=True)  # [1,D,H,W,3] in [-1,1]

        # Map from [-1,1] -> [0,1]
        grid_01 = (grid_m11 + 1.0) * 0.5

        return grid_01                            # [1,D,H,W,3]

    @staticmethod
    def _pad3x4_to4x4(theta3x4: torch.Tensor) -> torch.Tensor:
        """[N,3,4] -> [N,4,4] with last row [0,0,0,1]."""
        N = theta3x4.shape[0]
        eye = torch.eye(4, device=theta3x4.device, dtype=theta3x4.dtype).unsqueeze(0).repeat(N,1,1)
        eye[:, :3, :] = theta3x4
        return eye

    def _reconstruct_canonical(self) -> torch.Tensor:
        """Evaluate INR over the full canonical grid ([0,1]^3). Returns [D,H,W] (fp32)."""
        D, H, W = self.image_size
        coords = self._base_grid_01.to(self.device, dtype=torch.float32).view(-1, 3)  # [DHW,3]
        z = self.encoding(coords)                       # all points at once
        y = self.mlp(z).squeeze(-1)                     # [DHW]
        # blured_y = blur3d_separable(y.view(1,1,D, H, W).to(torch.float32), sigma=0.01, k=3)
        return y.view(D, H, W).to(torch.float32)

    # ---------- API ----------
    def forward(self, theta_m11_batch, dwi_stacks, stacks_indices, *, theta_is_canon2stack: bool = False):
        """
        theta_m11_batch : [N,3,4] or [N,4,4] in [-1,1].
                        If theta_is_canon2stack=True, it maps canonical→stack.
                        If False, it maps stack→canonical.
        dwi_stacks      : list of N tensors [D,H,W] (only for size/target loss outside)
        stacks_indices  : list of N index tensors/lists (which slices you compare)
        Returns:
        warped_stacks     : list of N [D,H,W] (canonical warped into each stack space)
        recon_canon       : [D,H,W] canonical reconstruction
        consistency_loss  : scalar tensor (variance across stacks back-warped to canonical)
        """
        # 1) Reconstruct canonical volume (full-grid on GPU)
        recon_canon = self._reconstruct_canonical()             # [D,H,W]
        vol = recon_canon.unsqueeze(0).unsqueeze(0)             # [1,1,D,H,W]

        # 2) Normalize theta to [N,4,4] on the correct device/dtype
        if theta_m11_batch.shape[-2:] == (3,4):
            theta44 = self._pad3x4_to4x4(theta_m11_batch.to(self.device, self.dtype))
        elif theta_m11_batch.shape[-2:] == (4,4):
            theta44 = theta_m11_batch.to(self.device, self.dtype)
        else:
            raise ValueError("theta must be [N,3,4] or [N,4,4] in [-1,1] space")

        theta_grid_s2c = theta44[:, :3, :]        # [N,3,4] stack[-1,1] -> canonical[-1,1]

        # 3) Warp canonical into each stack (for the data term outside)
        D, H, W = self.image_size
        warped_list = []
        for i in range(theta_grid_s2c.shape[0]):
            grid_s = F.affine_grid(theta_grid_s2c[i:i+1], size=(1,1,D,H,W), align_corners=True)  # [1,D,H,W,3]
            warped = F.grid_sample(vol, grid_s, mode='bilinear', padding_mode='zeros',
                                align_corners=True)  # [1,1,D,H,W]
            warped_list.append(warped[0,0])  # [D,H,W]

        # 4) Back-to-canonical consensus:
        #    Take each warped stack volume and map it back into canonical space
        #    using canonical->stack (the inverse of stack->canonical), then penalize
        #    disagreement across stacks in canonical space.
        T_c2s_44 = get_inverse_transformation(theta44)       # [N,4,4] canonical->stack
        theta_grid_c2s = T_c2s_44[:, :3, :]          # [N,3,4]

        canon_from_stacks = []
        for i in range(theta_grid_c2s.shape[0]):
            # Build grid for output canonical -> input stack
            grid_c = F.affine_grid(theta_grid_c2s[i:i+1], size=(1,1,D,H,W), align_corners=True)
            # Sample the stack prediction back to canonical
            stack_vol = warped_list[i].unsqueeze(0).unsqueeze(0)      # [1,1,D,H,W]
            back_canon = F.grid_sample(stack_vol, grid_c, mode='bilinear',
                                    padding_mode='reflection', align_corners=True)  # [1,1,D,H,W]
            canon_from_stacks.append(back_canon[0,0])                 # [D,H,W]

        canon_from_stacks = torch.stack(canon_from_stacks, dim=0)     # [N,D,H,W]
        consistency_loss = mask_aware_consistency(
            canon_from_stacks=canon_from_stacks,
            theta_grid_c2s=theta_grid_c2s,
            stacks_indices=stacks_indices,
            image_size=self.image_size,
            device=self.device,
            min_coverage=2
        )

        return warped_list, recon_canon, consistency_loss

    # def forward(self, theta_m11_batch, dwi_stacks, stacks_indices, *, theta_is_canon2stack: bool = False):
    #     """
    #     theta_m11_batch : [N,3,4] or [N,4,4] in [-1,1].
    #                       If theta_is_canon2stack=True, it maps canonical→stack.
    #                       If False, it maps stack→canonical.
    #     dwi_stacks      : list of N tensors [D,H,W] (only for size/target loss outside)
    #     stacks_indices  : list of N index tensors/lists (which slices you compare)
    #     Returns:
    #       warped_stacks : list of N [D,H,W] (canonical warped into each stack space)
    #       recon_canon   : [D,H,W] canonical reconstruction
    #     """
    #     # 1) Reconstruct canonical volume (full-grid on GPU)
    #     recon_canon = self._reconstruct_canonical()             # [D,H,W]
    #     vol = recon_canon.unsqueeze(0).unsqueeze(0)             # [1,1,D,H,W]

    #     # 2) Normalize theta to [N,4,4] on the correct device/dtype
    #     if theta_m11_batch.shape[-2:] == (3,4):
    #         theta44 = self._pad3x4_to4x4(theta_m11_batch.to(self.device, self.dtype))
    #     elif theta_m11_batch.shape[-2:] == (4,4):
    #         theta44 = theta_m11_batch.to(self.device, self.dtype)
    #     else:
    #         raise ValueError("theta must be [N,3,4] or [N,4,4] in [-1,1] space")

    #     # 3) affine_grid needs output[-1,1] -> input[-1,1] (stack -> canonical).
    #     #    If we were given canonical -> stack, invert once:
    #     if theta_is_canon2stack:
    #         theta44 = torch.linalg.inv(theta44)

    #     theta_grid = theta44[:, :3, :]            # [N,3,4]

    #     # 4) Warp canonical into each stack
    #     D, H, W = self.image_size
    #     warped_list = []
    #     for i in range(theta_grid.shape[0]):
    #         grid = F.affine_grid(theta_grid[i,...].unsqueeze(0), size=(1,1,D,H,W), align_corners=True)  # [1,D,H,W,3]
    #         warped = F.grid_sample(vol, grid, mode='bilinear', padding_mode='zeros',
    #                                align_corners=True)  # [1,1,D,H,W]
    #         warped_list.append(warped[0,0])  # [D,H,W]

    #     return warped_list, recon_canon

    @torch.no_grad()
    def get_recons(self):
        """Final volume on identity canonical grid."""
        self.eval()
        return self._reconstruct_canonical().to(torch.float32)
