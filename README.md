# Synapse-Vis: Generative 3D Synthetic Bone Scaffolds

A generative AI platform that creates 3D synthetic bone scaffold structures using a VAE-GAN architecture trained on bone-like microstructures.

## Pipeline

Micro-CT Bone Data → Preprocessing → 3D Voxel Grid → VAE + GAN → Generated Scaffold → Marching Cubes → STL Export → Three.js Viewer

## Tech Stack

- **PyTorch** — 3D CNN VAE + GAN training
- **NumPy / SciPy** — Voxel processing, biophysical metrics
- **scikit-image** — Marching cubes mesh extraction
- **Trimesh** — STL export
- **Flask** — Backend API
- **Three.js** — Interactive 3D scaffold viewer

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Project Structure

```
Synapse-vis/
├── app.py                  # Flask backend
├── data/                   # Data pipeline
│   ├── synthetic.py        # Synthetic training data generator
│   └── preprocess.py       # Micro-CT preprocessing
├── model/                  # ML models
│   ├── vae.py              # 3D VAE encoder-decoder
│   ├── gan.py              # GAN discriminator
│   ├── train.py            # Training loop
│   ├── generate.py         # Scaffold generation
│   └── metrics.py          # Biophysical metrics
├── geometry/               # Mesh processing
│   └── mesh_export.py      # Marching cubes + STL
├── static/                 # Frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
├── generated/              # Output STL files
├── checkpoints/            # Model weights
└── fallbacks/              # Pre-computed scaffolds
```

## License

MIT
