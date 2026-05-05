# Synapse-Vis: AI-Driven Generative Design for Biomimetic Bone Scaffolds

![Synapse-Vis Banner](https://img.shields.io/badge/Biomedical-AI-blueviolet?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Tech: PyTorch](https://img.shields.io/badge/AI-PyTorch-orange?style=for-the-badge)
![Tech: Three.js](https://img.shields.io/badge/3D-Three.js-black?style=for-the-badge)

**Synapse-Vis** is an advanced generative design platform that automates the creation of high-fidelity, biomimetic bone scaffolds for tissue engineering. By leveraging a **3D Variational Autoencoder (VAE)** trained on domain-informed soft-continuous fields, Synapse-Vis enables biomedical engineers to generate research-grade, 3D-printable architectures in seconds.

---

## The Challenge
Traditional bone scaffolds are designed using basic geometric primitives which frequently fail to replicate the **trabecular micro-architecture** of real bone. This leads to poor osseointegration (cell attachment) and mechanical failure. Manual design in CAD software is a massive bottleneck, often requiring days of engineering for a single optimized structure.

## The Solution: Synapse-Vis V2
Synapse-Vis serves as an AI-powered design suite that removes the mathematical friction from the design loop.
*   **Biomimetic Automation:** Sample from a learned latent space of TPMS (Triply Periodic Minimal Surface) architectures.
*   **Pixel-Perfect Fidelity:** Transitioned from binary voxels to **Soft-Continuous Fields**, achieving smooth organic curves suitable for clinical 3D printing.
*   **Biomedical Audit:** Automated validation of Porosity, Connectivity, Pore Diameter, and SA/V Ratio.

---

## Repository Architecture
A professional, modular codebase built for scalability and research integrity.

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

## Technical Methodology

### 1. Generative Strategy
We utilize a **3D-VAE** that maps complex bone architecture into a 128-dimensional latent space $Z$. The model is conditioned on target porosity $P$:

$$ \mathcal{L}_{total} = \mathcal{L}_{MSE} + \beta \mathcal{L}_{KL} + \lambda \mathcal{L}_{ADV} $$

### 2. From Pixels to Smooth Geometry
To solve the "staircase artifact" problem, we moved to **Soft-Field Learning**. Instead of predicting 0/1, the model predicts a continuous sigmoid gradient:
$$ f(V) = \sigma(k \cdot (G(x,y,z) - \tau)) $$
This allows **Marching Cubes** to extract a mathematically smooth isosurface at level 0.5.

### 3. Taubin Smoothing
We implement the non-shrinking **Taubin Filter** (30 iterations) to ensure organic curves without losing the volumetric integrity of the scaffold:
$$ X_{new} = (1 - \mu \nabla^2)(1 - \lambda \nabla^2) X_{old} $$

---

## Biomedical Metric Suite
Synapse-Vis performs a real-time clinical audit on every generated design:
*   **Connectivity:** CCL-based verification ensuring a 100% interconnected void network for nutrient transport.
*   **Pore Diameter:** Distance-transform based estimation, targeting the **100–500μm** clinical range.
*   **SA/V Ratio:** Surface-Area-to-Volume calculation to maximize cell attachment "real estate."

---

## Installation & Setup

1. **Clone & Setup**
   ```bash
   git clone https://github.com/Tanishi1/Synapse-vis.git
   cd Synapse-vis
   pip install -r requirements.txt
   ```

2. **Generate & Train (V2)**
   ```bash
   python data/synthetic.py
   python model/train.py
   ```

3. **Launch Platform**
   ```bash
   python app.py
   ```
   Visit `http://localhost:5000` to start designing.

---

## The Story: Analyzing Real Bone to Synthesize Solutions
We began by analyzing raw Micro-CT scans (V1). We realized that real-world medical data is often too noisy for direct AI training. We pivoted to **Domain-Informed Synthesis**, using clinical metrics from real bone to "calibrate" our mathematical training environment. This bridge between noisy reality and mathematical precision allowed us to achieve a **Reconstruction MSE of 0.00084** and true clinical-grade fidelity.

---

## Contributors
*   **OsteoForge AI**

---
> *"Engineering the Future of Bone Regeneration with Generative AI."*
