"""
3D Variational Autoencoder for bone scaffold generation.

Architecture:
  Encoder: 64³ voxel → 4³ feature map → μ, logσ (128-dim latent)
  Decoder: latent z + porosity_target → 64³ voxel reconstruction

Porosity conditioning: the target porosity (a single float 0-1)
is concatenated to z before the decoder. This lets you say
"generate a scaffold with 70% porosity" at inference time.
"""

import torch
import torch.nn as nn
import numpy as np


class Encoder3D(nn.Module):
    """
    Compresses a 64³ binary voxel grid into a 128-dim latent distribution.
    Uses four strided 3D convolutions: 64→32→16→8→4 spatial dims.
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 64³ → 32³
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32³ → 16³
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 16³ → 8³
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 8³ → 4³
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 256 channels × 4³ spatial = 16384 features
        self.flat_dim = 256 * 4 * 4 * 4

        # Two separate heads: one for mean, one for log-variance
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        """x: (B, 1, 64, 64, 64) → mu: (B, 128), logvar: (B, 128)"""
        h = self.conv_layers(x)          # (B, 256, 4, 4, 4)
        h = h.flatten(start_dim=1)       # (B, 16384)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder3D(nn.Module):
    """
    Reconstructs a 64³ voxel grid from latent z + porosity target.
    Input: z (128) concatenated with porosity (1) = 129 values.
    Uses four transposed 3D convolutions: 4→8→16→32→64 spatial dims.
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        # +1 for porosity conditioning
        self.fc = nn.Linear(latent_dim + 1, 256 * 4 * 4 * 4)

        self.deconv_layers = nn.Sequential(
            # 4³ → 8³
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # 8³ → 16³
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # 16³ → 32³
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 32³ → 64³ — Sigmoid maps output to [0, 1] (probability of being solid)
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, porosity):
        """
        z: (B, 128), porosity: (B,) scalar in [0, 1]
        Returns: (B, 1, 64, 64, 64)
        """
        # Concatenate porosity as an extra conditioning dimension
        p = porosity.unsqueeze(1)               # (B, 1)
        z_cond = torch.cat([z, p], dim=1)       # (B, 129)
        h = self.fc(z_cond)                     # (B, 16384)
        h = h.view(-1, 256, 4, 4, 4)           # (B, 256, 4, 4, 4)
        return self.deconv_layers(h)            # (B, 1, 64, 64, 64)


class BoneVAE(nn.Module):
    """
    Full VAE: encoder → reparameterize → decoder.
    Training: forward() returns reconstruction + distribution params for loss.
    Inference: generate() samples from prior and decodes with target porosity.
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)

    def reparameterize(self, mu, logvar):
        """
        The reparameterization trick: sample z = mu + eps * sigma.
        During training: stochastic (eps ~ N(0,1)).
        During eval: deterministic (uses mu directly).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, porosity):
        """
        x: (B, 1, 64, 64, 64), porosity: (B,)
        Returns: reconstruction (B, 1, 64, 64, 64), mu, logvar
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, porosity)
        return recon, mu, logvar

    def generate(self, porosity_value: float, device: str = "cpu") -> np.ndarray:
        """
        Generate one scaffold voxel grid with the given target porosity.
        porosity_value: float in [0, 1], e.g. 0.70 for 70% pore space.
        Returns: binary numpy array (64, 64, 64), 1=solid, 0=pore.
        """
        self.eval()
        with torch.no_grad():
            # Sample from standard normal prior
            z = torch.randn(1, self.latent_dim).to(device)
            # Porosity conditioning
            p = torch.tensor([porosity_value], dtype=torch.float32).to(device)
            # Decode
            voxel_prob = self.decoder(z, p)                          # (1, 1, 64, 64, 64)
            voxel_np = voxel_prob.squeeze().cpu().numpy()            # (64, 64, 64)
            # Threshold at 0.5: solid if probability > 0.5
            return (voxel_np > 0.5).astype(np.float32)
