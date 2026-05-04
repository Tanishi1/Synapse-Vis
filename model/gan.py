"""
3D GAN Discriminator for adversarial training of the VAE decoder.

Why spectral normalization?
  Regular batch norm in discriminators causes training instability
  because the discriminator can grow too powerful too quickly.
  Spectral norm constrains the Lipschitz constant of each layer,
  making training smoother without needing careful hyperparameter tuning.

The discriminator sees a 64³ voxel grid and outputs a single scalar:
  1.0 → this looks like real bone microstructure
  0.0 → this looks generated/fake

It does NOT know whether it's looking at real or generated data — that
label comes from the training loop. The competition between discriminator
(trying to detect fakes) and VAE decoder (trying to fool it) is what
forces the decoder to produce sharp, realistic-looking structures.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator3D(nn.Module):
    """
    Four-layer 3D discriminator with spectral normalization.
    No BatchNorm here — it interacts badly with spectral norm.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 64³ → 32³
            spectral_norm(nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 32³ → 16³
            spectral_norm(nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 16³ → 8³
            spectral_norm(nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 8³ → 4³
            spectral_norm(nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 × 4³ = 16384 → 1 scalar
            nn.Flatten(),
            spectral_norm(nn.Linear(256 * 4 * 4 * 4, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (B, 1, 64, 64, 64) → (B, 1) real/fake probability"""
        return self.net(x)
