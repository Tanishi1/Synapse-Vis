"""
VAE + GAN training loop — updated for soft-continuous gyroid training data.
 
KEY CHANGES from previous version:
  1. Reconstruction loss: BCE → MSE
     BCE penalises values near 0.5 heavily (treats them as wrong predictions).
     MSE penalises proportionally to distance — correct for continuous [0,1] fields.
 
  2. KL weight beta: 0.005 (unchanged, works well for structural data)
 
  3. Adversarial weight lambda: 0.05 (reduced from 0.1)
     The discriminator should help sharpen surfaces, not dominate the loss.
     With continuous data the reconstruction signal is already strong.
 
  4. Epochs: 200 (was 150) — continuous data is richer, benefits from more training.
     Set to 100 if you want a faster first run to check mesh quality.
 
Loss breakdown:
  L_recon : MSE(decoded_output, real_input)
            Teaches decoder to reproduce smooth gyroid structure faithfully.
 
  L_kl    : KL divergence between N(mu, sigma) and N(0,1).
            Keeps latent space smooth for interpolation/sampling.
 
  L_adv   : VAE decoder tries to fool the discriminator.
            Sharpens pore surfaces and removes blurriness.
 
  L_disc  : BCE(real→1) + BCE(fake→0) for the discriminator.
 
Run: python model/train.py
"""
 
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vae import BoneVAE
from model.gan import Discriminator3D
 
 
class BoneDataset(Dataset):
    def __init__(self, data_dir="data/processed"):
        patches_path = os.path.join(data_dir, "patches.npy")
        porosities_path = os.path.join(data_dir, "porosities.npy")
 
        if not os.path.exists(patches_path):
            raise FileNotFoundError(
                f"{patches_path} not found.\n"
                "Run: python data/synthetic.py"
            )
 
        self.patches = np.load(patches_path)        # (N, 64, 64, 64) float32 in [0,1]
        self.porosities = np.load(porosities_path)  # (N,) float32
 
        # Sanity check: confirm this is soft data, not binary
        unique_approx = len(np.unique(np.round(self.patches[:5].ravel(), 1)))
        if unique_approx <= 3:
            print("WARNING: Dataset looks binary. Re-run data/synthetic.py for soft fields.")
        else:
            print(f"✓ Soft-field dataset confirmed — {unique_approx}+ distinct values per sample.")
 
        print(f"Dataset: {len(self.patches)} samples | "
              f"porosity {self.porosities.min()*100:.0f}–{self.porosities.max()*100:.0f}%")
 
    def __len__(self):
        return len(self.patches)
 
    def __getitem__(self, idx):
        # (64,64,64) → (1,64,64,64)
        voxel = torch.FloatTensor(self.patches[idx]).unsqueeze(0)
        porosity = torch.FloatTensor([self.porosities[idx]])
        return voxel, porosity
 
 
def train(
    data_dir="data/processed",
    checkpoint_dir="checkpoints",
    epochs=100,
    batch_size=8,
    lr_vae=1e-4,
    lr_disc=4e-5,
    beta=0.005,      # KL weight — keep low to preserve structural detail
    lam=0.05,        # adversarial weight — lower than before; MSE recon is already strong
    save_every=20,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} | VRAM: {mem:.1f} GB")
    else:
        print("WARNING: No GPU detected. Training on CPU will take 4–6 hours.")
 
    dataset = BoneDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
 
    vae = BoneVAE(latent_dim=128).to(device)
    disc = Discriminator3D().to(device)
 
    opt_vae = torch.optim.Adam(vae.parameters(), lr=lr_vae, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))
 
    # Cosine annealing: smoothly reduces LR over training
    sched_vae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vae, T_max=epochs, eta_min=1e-6)
    sched_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=epochs, eta_min=1e-7)
 
    bce = nn.BCELoss()
 
    # Resume from checkpoint if it exists
    start_epoch = 0
    final_ckpt = os.path.join(checkpoint_dir, "model_final.pth")
    if os.path.exists(final_ckpt):
        print(f"\nFound existing checkpoint — loading {final_ckpt}")
        print("Delete checkpoints/model_final.pth to train from scratch.")
        state = torch.load(final_ckpt, map_location=device)
        vae.load_state_dict(state["vae"])
        disc.load_state_dict(state["disc"])
        start_epoch = state.get("epoch", 0)
        print(f"Resuming from epoch {start_epoch}")
 
    print(f"\nStarting training: {epochs} epochs | batch_size={batch_size} | loss=MSE+KL+ADV")
    print("=" * 65)
 
    best_recon = float("inf")
 
    for epoch in range(start_epoch, epochs):
        vae.train()
        disc.train()
 
        total_recon = total_kl = total_adv = total_disc = 0.0
 
        for batch_voxels, batch_porosity in loader:
            batch_voxels = batch_voxels.to(device)              # (B, 1, 64, 64, 64)
            batch_porosity = batch_porosity.squeeze(1).to(device)  # (B,)
 
            # ── Step 1: Train Discriminator ──────────────────────────────
            with torch.no_grad():
                recon, _, _ = vae(batch_voxels, batch_porosity)
 
            real_preds = disc(batch_voxels)
            fake_preds = disc(recon.detach())
 
            # Label smoothing: 0.9 / 0.1 instead of 1.0 / 0.0
            real_labels = torch.full_like(real_preds, 0.9)
            fake_labels = torch.full_like(fake_preds, 0.1)
 
            d_loss = bce(real_preds, real_labels) + bce(fake_preds, fake_labels)
 
            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()
 
            # ── Step 2: Train VAE ────────────────────────────────────────
            recon, mu, logvar = vae(batch_voxels, batch_porosity)
 
            # MSE reconstruction loss — correct for continuous [0,1] fields.
            # Measures mean squared pixel-wise difference.
            recon_loss = F.mse_loss(recon, batch_voxels)
 
            # KL divergence — regularises latent space
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
 
            # Adversarial — sharpen mesh surfaces
            fake_preds_vae = disc(recon)
            adv_loss = bce(fake_preds_vae, torch.ones_like(fake_preds_vae))
 
            vae_loss = recon_loss + beta * kl_loss + lam * adv_loss
 
            opt_vae.zero_grad()
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            opt_vae.step()
 
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_adv += adv_loss.item()
            total_disc += d_loss.item()
 
        sched_vae.step()
        sched_disc.step()
 
        n = len(loader)
        avg_recon = total_recon / n
        avg_kl    = total_kl / n
        avg_adv   = total_adv / n
        avg_disc  = total_disc / n
 
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"Recon(MSE):{avg_recon:.5f}  KL:{avg_kl:.3f}  "
                f"Adv:{avg_adv:.3f}  Disc:{avg_disc:.3f} | "
                f"LR:{sched_vae.get_last_lr()[0]:.2e}"
            )
 
        # Save best checkpoint by reconstruction loss
        if avg_recon < best_recon:
            best_recon = avg_recon
            torch.save(
                {"vae": vae.state_dict(), "disc": disc.state_dict(),
                 "epoch": epoch + 1, "recon_loss": avg_recon},
                os.path.join(checkpoint_dir, "model_best.pth"),
            )
 
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save(
                {"vae": vae.state_dict(), "disc": disc.state_dict(), "epoch": epoch + 1},
                ckpt_path,
            )
            print(f"  ✓ Checkpoint saved: {ckpt_path}")
 
    # Save final model
    torch.save(
        {"vae": vae.state_dict(), "disc": disc.state_dict(), "epoch": epochs},
        final_ckpt,
    )
    print(f"\n✓ Training complete. Final model saved: {final_ckpt}")
    print(f"  Best reconstruction MSE: {best_recon:.5f}")
    print(f"  (Lower = better. Good target: < 0.008)")
 
    _pregenerate_fallbacks(vae, device)
 
 
def _pregenerate_fallbacks(vae, device, out_dir="fallbacks"):
    """Pre-generate 5 scaffolds at different porosity levels for demo safety."""
    import json
    from model.metrics import compute_scaffold_metrics
    from geometry.mesh_export import voxel_to_stl
 
    os.makedirs(out_dir, exist_ok=True)
    vae.eval()
    targets = [0.60, 0.65, 0.70, 0.75, 0.80]
    print("\nPre-generating fallback scaffolds...")
 
    for pt in targets:
        voxel = vae.generate(pt, device=device)
        metrics = compute_scaffold_metrics(voxel)
        key = int(pt * 100)
        stl_path = os.path.join(out_dir, f"scaffold_{key}.stl")
        voxel_to_stl(voxel, stl_path)
        metrics_path = os.path.join(out_dir, f"metrics_{key}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        print(f"  {key}% → porosity={metrics['porosity_pct']:.1f}%  viable={metrics['biologically_viable']}")
 
    print("✓ Fallbacks ready.")
 
 
if __name__ == "__main__":
    train()
