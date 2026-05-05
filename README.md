# Synapse-Vis: AI-Driven Generative Design for Biomimetic Bone Scaffolds

![Synapse-Vis Banner](https://img.shields.io/badge/Biomedical-AI-blueviolet?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Tech: PyTorch](https://img.shields.io/badge/AI-PyTorch-orange?style=for-the-badge)
![Tech: Three.js](https://img.shields.io/badge/3D-Three.js-black?style=for-the-badge)

> **Important Notice:** The `v2` branch contains the finalized, clinical-grade generative engine and is the official product version of Synapse-Vis.

**Synapse-Vis** is an advanced generative design platform that automates the creation of high-fidelity, biomimetic bone scaffolds for tissue engineering. By leveraging a **3D Variational Autoencoder (VAE)** trained on domain-informed soft-continuous fields, Synapse-Vis enables biomedical engineers to generate research-grade, 3D-printable architectures in seconds.

---

## The Challenge
Traditional bone scaffolds designed using basic geometric primitives frequently fail to replicate the complex **trabecular micro-architecture** of real bone. This leads to poor osseointegration and mechanical failure. Manual CAD design is a massive bottleneck in biomedical research; Synapse-Vis solves this by automating the generation of complex, interconnected architectures.

## The Solution: Synapse-Vis V2
The V2 pipeline represents a leap in generative precision:
*   **Biomimetic Automation:** Generates TPMS (Triply Periodic Minimal Surface) architectures optimized for bone growth.
*   **Soft-Field Fidelity:** Uses continuous sigmoid gradients instead of binary voxels to achieve organic, smooth surfaces.
*   **Biomedical Audit:** Automated validation of Porosity, Connectivity, Pore Diameter, and SA/V Ratio.

---

## Repository Architecture
```text
Synapse-Vis/
├── app.py                  # Flask Backend & AI Inference API
├── requirements.txt        # Production Dependencies
├── data/                   # Data Pipeline
│   ├── synthetic.py        # Soft-Field Gyroid Generator (V2 Engine)
│   └── processed/          # Training Data Storage
├── model/                  # Deep Learning Core
│   ├── vae.py              # 3D Variational Autoencoder (Volumetric)
│   ├── gan.py              # Discriminator for structural sharpening
│   ├── train.py            # VAE+GAN Training Loop (MSE Loss)
│   └── metrics.py          # Biomedical Verification Engine (EDT/CCL)
├── geometry/               # Computational Geometry
│   ├── mesh_export.py      # Voxel-to-STL (Marching Cubes + Taubin)
│   └── cylinder_mask.py    # Clinical Plug Masking logic
├── static/                 # Frontend Assets
│   ├── index.html          # Dashboard UI
│   ├── app.js              # Three.js 3D Viewer & API Controller
│   └── styles.css          # Glassmorphic Design System
├── checkpoints/            # Trained Model Weights (.pth)
└── generated/              # Temporary STL Cache for Export
```

---

## Data Flow & Pipeline Architecture

### 1. Training Pipeline (Offline)
The pipeline begins by analyzing real Micro-CT bone parameters to calibrate a mathematical "Synthetic Twin" generator.
1. **Data Generation:** `synthetic.py` produces 800 soft-continuous 3D fields.
2. **AI Training:** `train.py` uses MSE loss to train the VAE to reconstruct these fields with high structural fidelity ($MSE < 0.001$).
3. **Weights:** Final weights are stored in `checkpoints/model_final.pth`.

### 2. Inference & Generation (Live)
When a user requests a scaffold:
1. **VAE Sampling:** The decoder samples a 3D field from the latent space based on target porosity.
2. **Masking:** A cylindrical plug mask with smooth `tanh` boundaries is applied.
3. **Metric Audit:** The system verifies topological connectivity and porosity in real-time.
4. **Mesh Extraction:** Marching Cubes extracts an isosurface at level 0.5, followed by 30 iterations of **Taubin Smoothing**.
5. **Visualization:** The resulting STL is rendered via **Three.js** for interactive inspection.

---

## Technical Methodology

### Generative Strategy
We utilize a **3D-VAE** that maps complex bone architecture into a 128-dimensional latent space $Z$. The model is conditioned on target porosity $P$:
$$ \mathcal{L}_{total} = \mathcal{L}_{MSE} + \beta \mathcal{L}_{KL} + \lambda \mathcal{L}_{ADV} $$

### Non-Shrinking Smoothing
To ensure organic curves without losing volumetric integrity, we implement the **Taubin Filter**:
$$ X_{new} = (1 - \mu \nabla^2)(1 - \lambda \nabla^2) X_{old} $$

---

## Installation & Execution Instructions

Follow these steps to run the finalized V2 product:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Tanishi1/Synapse-vis.git
   cd Synapse-vis
   ```

2. **Switch to the V2 Branch (Required)**
   ```bash
   git checkout v2
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize V2 Data & Training (Optional - Pre-trained weights included)**
   ```bash
   python data/synthetic.py
   python model/train.py
   ```

5. **Start the Platform**
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` to begin designing.

---

## Contributors
*   **OsteoForge AI**

---
> *"Engineering the Future of Bone Regeneration with Generative AI."*
