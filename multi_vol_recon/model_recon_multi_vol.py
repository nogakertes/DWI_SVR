import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from contextlib import nullcontext
import sys
sys.path.append('.')
from utils import *


class ReconNet_canonial_grid(nn.Module):
    """
    This model reconstructs multiple canonical volumes (one per "vol") simultaneously.
    - INR_networks is a module list of networks, one per canonical volume.
    - In the forward, inputs are:
        - theta_m11_batch: list of [N_stacks, 3, 4] or [N_stacks, 4, 4], length n_vols
        - dwi_stacks: list of lists; outer len n_vols, inner list contains N_stacks [D,H,W] tensors
        - stacks_indices: list of lists; outer len n_vols, inner list contains indices for each stack in that vol
    The model produces stacked results for each volume.
    """

    def __init__(self, recon_config, image_size, n_vols, device='cuda', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype  = dtype
        self.image_size = tuple(image_size)  # (D,H,W)
        self.n_vols = n_vols

        # tcnn: coords ∈ [0,1]^3
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=recon_config["encoding"]
        ).to(device=device, dtype=dtype)

        feat_dim = self.encoding.n_output_dims
        
        self.INR_networks = nn.ModuleList([
            tcnn.Network(
                n_input_dims=feat_dim,
                n_output_dims=1,
                network_config=recon_config["network"]
            ).to(device=device, dtype=dtype)
            for _ in range(n_vols)
        ])

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
        theta = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0)  # [1,3,4]
        grid_m11 = F.affine_grid(theta, size=(1,1,D,H,W), align_corners=True)  # [1,D,H,W,3] in [-1,1]
        grid_01 = (grid_m11 + 1.0) * 0.5
        return grid_01                            # [1,D,H,W,3]

    @staticmethod
    def _pad3x4_to4x4(theta3x4: torch.Tensor) -> torch.Tensor:
        """[N,3,4] -> [N,4,4] with last row [0,0,0,1]."""
        N = theta3x4.shape[0]
        eye = torch.eye(4, device=theta3x4.device, dtype=theta3x4.dtype).unsqueeze(0).repeat(N,1,1)
        eye[:, :3, :] = theta3x4
        return eye

    def _reconstruct_canonical(self, vol_idx: int) -> torch.Tensor:
        """
        Evaluate INR for canonical grid for vol index vol_idx.
        Returns [D,H,W] (fp32).
        """
        D, H, W = self.image_size
        coords = self._base_grid_01.to(self.device, dtype=torch.float32).view(-1, 3)  # [DHW,3]
        z = self.encoding(coords)                       # [DHW, feat_dim]
        y = self.INR_networks[vol_idx](z).squeeze(-1)   # [DHW]
        return y.view(D, H, W).to(torch.float32)

    # ---------- API ----------
    def forward(self, theta_m11_batch, dwi_stacks, stacks_indices, *, theta_is_canon2stack: bool = False):
        """
        theta_m11_batch: list of n_vols tensors [N_stacks,3,4] or [N_stacks,4,4] in [-1,1] (per volume/study)
        dwi_stacks: list of n_vols lists, each contains N_stacks [D,H,W] tensors
        stacks_indices: list of n_vols lists, each contains stack index lists/tensors
        Returns:
        warped_stacks: list of n_vols lists, each [N_stacks, D,H,W] warped canonical volumes
        recon_canons:  list of n_vols [D,H,W] canonical reconstructions
        consistency_losses: list of n_vols scalars (per-volume consensus)
        """
        D, H, W = self.image_size
        n_vols = self.n_vols
        warped_stacks_all = []
        recon_canons_all = []
        consistency_losses_all = []

        for v in range(n_vols):
            # 1) Reconstruct canonical volume for volume v
            recon_canon = self._reconstruct_canonical(v)         # [D,H,W]
            vol = recon_canon.unsqueeze(0).unsqueeze(0)          # [1,1,D,H,W]
            recon_canons_all.append(recon_canon)

            # 2) Normalize theta for this volume
            theta_input = theta_m11_batch[v]
            if theta_input.shape[-2:] == (3,4):
                theta44 = self._pad3x4_to4x4(theta_input.to(self.device, self.dtype))
            elif theta_input.shape[-2:] == (4,4):
                theta44 = theta_input.to(self.device, self.dtype)
            else:
                raise ValueError("theta must be [N,3,4] or [N,4,4] in [-1,1] space")

            theta_grid_s2c = theta44[:, :3, :]        # [N,3,4] stack[-1,1] -> canonical[-1,1]
            # 3) Warp canonical into each stack (for the data term outside)
            stacks_for_vol = dwi_stacks[v]
            warped_list = []
            N_stacks = theta_grid_s2c.shape[0]  # number of stacks for this volume
            for i in range(N_stacks):
                grid_s = F.affine_grid(theta_grid_s2c[i:i+1], size=(1,1,D,H,W), align_corners=True)  # [1,D,H,W,3]
                warped = F.grid_sample(
                    vol, grid_s, mode='bilinear', padding_mode='zeros',
                    align_corners=True
                )  # [1,1,D,H,W]
                warped_list.append(warped[0,0])       # [D,H,W]
            warped_stacks_all.append(torch.stack(warped_list))     # list of [N_stacks, D,H,W]

            # 4) Back-to-canonical consensus (per volume)
            T_c2s_44 = get_inverse_transformation(theta44)       # [N,4,4] canonical->stack
            theta_grid_c2s = T_c2s_44[:, :3, :]          # [N,3,4]
            canon_from_stacks = []
            for i in range(N_stacks):
                grid_c = F.affine_grid(theta_grid_c2s[i:i+1], size=(1,1,D,H,W), align_corners=True)
                stack_vol = warped_list[i].unsqueeze(0).unsqueeze(0)      # [1,1,D,H,W]
                back_canon = F.grid_sample(
                    stack_vol, grid_c, mode='bilinear',
                    padding_mode='reflection', align_corners=True
                )  # [1,1,D,H,W]
                canon_from_stacks.append(back_canon[0,0])                 # [D,H,W]
            canon_from_stacks = torch.stack(canon_from_stacks, dim=0)     # [N_stacks, D,H,W]

            # stacks_indices[v] is the corresponding indices for vol v
            consistency_loss = mask_aware_consistency(
                canon_from_stacks=canon_from_stacks,
                theta_grid_c2s=theta_grid_c2s,
                stacks_indices=stacks_indices,
                image_size=self.image_size,
                device=self.device,
                min_coverage=2
            )
            consistency_losses_all.append(consistency_loss)

        return warped_stacks_all, recon_canons_all, torch.mean(torch.stack(consistency_losses_all))

    @torch.no_grad()
    def get_recons(self):
        """Final volumes for all canonical grids ([D,H,W] x n_vols)"""
        self.eval()
        recons = []
        for v in range(self.n_vols):
            recons.append(self._reconstruct_canonical(v).to(torch.float32))
        return recons

