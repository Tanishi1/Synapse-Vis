"""
Combined VAE + GAN training loop.

Loss breakdown:
  L_recon   : Binary cross-entropy between decoded output and real input.
               Encourages the decoder to faithfully reproduce bone structure.

  L_kl      : KL divergence between learned latent distribution and N(0,1).
               Keeps the latent space smooth so sampling at inference works.
               Weighted by beta (small value like 0.005 — too large makes
               the model ignore structure, too small means sampling gives noise).

  L_adv     : The VAE decoder trying to fool the discriminator.
               This is what sharpens the output — the discriminator penalizes
               any blurry or unrealistic voxel patterns.

  Discriminator is trained separately with:
  L_disc = BCE(real → 1) + BCE(generated → 0)

Run: python model/train.py
"""

import os
import sys
import torch
import torch.nn as nn
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
                "Run: python data/download_data.py  then  python data/preprocess.py"
            )

        self.patches = np.load(patches_path)      # (N, 64, 64, 64)
        self.porosities = np.load(porosities_path)  # (N,)

        print(f"Dataset loaded: {len(self.patches)} patches")
        print(f"Porosity range: {self.porosities.min()*100:.1f}% — {self.porosities.max()*100:.1f}%")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Add channel dim: (64,64,64) → (1,64,64,64)
        voxel = torch.FloatTensor(self.patches[idx]).unsqueeze(0)
        porosity = torch.FloatTensor([self.porosities[idx]])
        return voxel, porosity


def train(
    data_dir="data/processed",
    checkpoint_dir="checkpoints",
    epochs=150,
    batch_size=8,
    lr_vae=1e-4,
    lr_disc=5e-5,
    beta=0.005,       # KL weight — keep small to preserve structure detail
    lam=0.1,          # adversarial weight — increase if outputs look blurry
    save_every=10,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    if device == "cpu":
        print("WARNING: CPU training is slow. Expect ~3-4 hours for 150 epochs.")
        print("Reduce epochs to 50 for a quick hackathon demo run.")

    # Data
    dataset = BoneDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Models
    vae = BoneVAE(latent_dim=128).to(device)
    disc = Discriminator3D().to(device)

    # Optimizers — discriminator gets a lower learning rate to avoid
    # it becoming too powerful too quickly (training instability)
    opt_vae = torch.optim.Adam(vae.parameters(), lr=lr_vae, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.5, 0.999))

    # LR scheduler: halve LR every 50 epochs to fine-tune later
    sched_vae = torch.optim.lr_scheduler.StepLR(opt_vae, step_size=50, gamma=0.5)
    sched_disc = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=50, gamma=0.5)

    bce = nn.BCELoss()

    # Resume from checkpoint if exists
    start_epoch = 0
    final_ckpt = os.path.join(checkpoint_dir, "model_final.pth")
    if os.path.exists(final_ckpt):
        print(f"Found existing checkpoint — loading {final_ckpt}")
        state = torch.load(final_ckpt, map_location=device)
        vae.load_state_dict(state["vae"])
        disc.load_state_dict(state["disc"])
        start_epoch = state.get("epoch", 0)
        print(f"Resuming from epoch {start_epoch}")

    print(f"\nStarting training: {epochs} epochs, batch_size={batch_size}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        vae.train()
        disc.train()

        total_vae_loss = 0.0
        total_disc_loss = 0.0

        for batch_voxels, batch_porosity in loader:
            batch_voxels = batch_voxels.to(device)         # (B, 1, 64, 64, 64)
            batch_porosity = batch_porosity.squeeze(1).to(device)  # (B,)

            # ─── Step 1: Train Discriminator ─────────────────────────
            # Get VAE reconstruction (no grad needed for VAE here)
            with torch.no_grad():
                recon, _, _ = vae(batch_voxels, batch_porosity)

            real_preds = disc(batch_voxels)
            fake_preds = disc(recon)

            # Real samples → discriminator should output 1
            # Fake samples → discriminator should output 0
            real_labels = torch.ones_like(real_preds) * 0.9  # label smoothing
            fake_labels = torch.zeros_like(fake_preds) + 0.1  # label smoothing

            d_loss = bce(real_preds, real_labels) + bce(fake_preds, fake_labels)

            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            # ─── Step 2: Train VAE (encoder + decoder) ───────────────
            recon, mu, logvar = vae(batch_voxels, batch_porosity)

            # Reconstruction: how well did we reproduce the input?
            recon_loss = bce(recon, batch_voxels)

            # KL divergence: keep latent space well-organized
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )

            # Adversarial: decoder tries to fool discriminator
            # (treats generated output as if it were real)
            fake_preds_for_vae = disc(recon)
            adv_loss = bce(fake_preds_for_vae, torch.ones_like(fake_preds_for_vae))

            vae_loss = recon_loss + beta * kl_loss + lam * adv_loss

            opt_vae.zero_grad()
            vae_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            opt_vae.step()

            total_vae_loss += vae_loss.item()
            total_disc_loss += d_loss.item()

        sched_vae.step()
        sched_disc.step()

        avg_vae = total_vae_loss / len(loader)
        avg_disc = total_disc_loss / len(loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"VAE: {avg_vae:.4f} | "
                f"Disc: {avg_disc:.4f} | "
                f"LR: {sched_vae.get_last_lr()[0]:.2e}"
            )

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save(
                {"vae": vae.state_dict(), "disc": disc.state_dict(), "epoch": epoch + 1},
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

    # Save final model
    torch.save(
        {"vae": vae.state_dict(), "disc": disc.state_dict(), "epoch": epochs},
        final_ckpt,
    )
    print(f"\nTraining complete. Model saved: {final_ckpt}")

    # Pre-generate fallback scaffolds for demo safety
    _pregenerate_fallbacks(vae, device)


def _pregenerate_fallbacks(vae, device, out_dir="fallbacks"):
    """
    Pre-generate 5 scaffolds at different porosity levels.
    These are served instantly if live generation fails during the demo.
    """
    import json
    import sys
    sys.path.insert(0, ".")
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

        print(f"  {key}% porosity → {metrics['porosity_pct']:.1f}% actual, viable={metrics['biologically_viable']}")

    print("Fallbacks ready.")


if __name__ == "__main__":
    train()
