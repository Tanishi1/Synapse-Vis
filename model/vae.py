import torch
import torch.nn as nn
import numpy as np
 
 
class Encoder3D(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = 256 * 4 * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
 
    def forward(self, x):
        h = self.conv_layers(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
 
 
class Decoder3D(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim + 1, 256 * 4 * 4 * 4)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # output in [0, 1]
        )
 
    def forward(self, z, porosity):
        p = porosity.unsqueeze(1)
        z_cond = torch.cat([z, p], dim=1)
        h = self.fc(z_cond).view(-1, 256, 4, 4, 4)
        return self.deconv_layers(h)
 
 
class BoneVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder3D(latent_dim)
        self.decoder = Decoder3D(latent_dim)
 
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu
 
    def forward(self, x, porosity):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, porosity)
        return recon, mu, logvar
 
    def generate(self, porosity_value: float, device: str = "cpu") -> np.ndarray:
        """
        Sample z ~ N(0,I), decode with target porosity.
        Returns SOFT CONTINUOUS field [0,1] — not thresholded to binary.
        voxel_to_stl() runs marching cubes at 0.5 on this field.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim).to(device)
            p = torch.tensor([porosity_value], dtype=torch.float32).to(device)
            voxel_prob = self.decoder(z, p)
            voxel_np = voxel_prob.squeeze().cpu().numpy()
            # ← KEY: return soft output, NOT (voxel_np > 0.5)
            return voxel_np.astype(np.float32)
