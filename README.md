# Synapse-Vis: AI-Driven Generative Design for Biomimetic Bone Scaffolds

![Synapse-Vis Banner](https://img.shields.io/badge/Biomedical-AI-blueviolet?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Tech: PyTorch](https://img.shields.io/badge/AI-PyTorch-orange?style=for-the-badge)
![Tech: Three.js](https://img.shields.io/badge/3D-Three.js-black?style=for-the-badge)

> **Official Product Branch:** The `v2` branch contains the finalized, clinical-grade generative engine.

---

## Problem Statement
In the field of regenerative medicine, the successful integration of bone implants (osseointegration) depends almost entirely on the **micro-architecture** of the scaffold. 

Currently, biomedical engineers face three critical bottlenecks:
1.  **CAD Limitations:** Traditional CAD software is designed for mechanical parts, not biological structures. Creating complex, interconnected "Gyroid" lattices manually is mathematically exhausting and slow.
2.  **Clinical Disconnect:** Many generated scaffolds look good visually but lack the **interconnected porosity** required for nutrient transport and blood vessel formation (angiogenesis).
3.  **The Iteration Gap:** Validating a design for porosity and connectivity currently requires days of simulation. Researchers need a way to generate and validate "research-grade" candidates in seconds, not hours.

## The Solution: Synapse-Vis
Synapse-Vis is an AI-powered design automation suite that bridges the gap between **Generative Deep Learning** and **Computational Geometry**. We provide an automated pipeline that allows engineers to instantly generate, visualize, and audit biomimetic scaffolds that are 3D-print ready.

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

## Technical Methodology & Data Flow

### 1. Domain-Informed Data Generation
The V2 engine uses a **"Synthetic Twin"** strategy. We analyzed the biological parameters of real trabecular bone (pore size, strut density) to calibrate a mathematical **Triply Periodic Minimal Surface (TPMS)** generator. This generator produces **Soft-Continuous Fields** (values 0.0–1.0) instead of binary pixels, allowing the AI to learn smooth surface gradients.

### 2. Generative Engine (VAE + GAN)
We utilize a **3D Variational Autoencoder** to map the design space into a 128-dimensional latent manifold. 
*   **MSE Loss:** Used to ensure the AI reproduces the smooth gradients of the gyroid lattice faithfully.
*   **Adversarial Training:** A 3D Discriminator ensures that the pore-walls are sharp and structurally sound, eliminating the "blurriness" common in standard VAEs.

### 3. Verification & Geometry
*   **Topological Audit:** Using **Connected Components Labeling (CCL)**, the system verifies that every pore is part of a single interconnected network.
*   **Organic Smoothing:** We apply 30 iterations of the **Taubin Filter**, a non-shrinking smoothing algorithm that preserves the targeted porosity while creating organic, bone-like curves.

---

## Clinical Significance
By automating the design of **Gyroid Lattices**, Synapse-Vis targets the "Clinical Sweet Spot":
*   **Pore Diameter (100–500μm):** The range required for vascularization.
*   **Porosity (55–85%):** Optimized for bone-ingrowth vs. mechanical strength.
*   **High SA/V Ratio:** Maximizing the surface area for osteoblast cell attachment.

---

## Installation & Execution

1. **Clone & Setup**
   ```bash
   git clone https://github.com/Tanishi1/Synapse-vis.git
   cd Synapse-vis
   git checkout v2
   pip install -r requirements.txt
   ```

2. **Initialization** (Optional - pre-trained weights are provided in `checkpoints/`)
   ```bash
   python data/synthetic.py
   python model/train.py
   ```

3. **Run Platform**
   ```bash
   python app.py
   ```

---

## The Story: Bridging Noise and Precision
We started by analyzing raw Micro-CT data (V1), but found it too noisy for clinical-grade AI. We pivoted to a **Domain-Informed Synthesis** model, using real clinical metrics to build a "perfect" training environment. This breakthrough allowed us to achieve a **Reconstruction MSE of 0.00084**, delivering a tool that combines the organic complexity of biology with the mathematical precision of engineering.

---

## Contributors
*   **OsteoForge AI**

---
> *"Engineering the Future of Bone Regeneration with Generative AI."*
