"""
Flask backend for Synapse-Vis.

Endpoints:
  GET  /               → serves index.html
  POST /api/generate   → generates scaffold, returns metrics + job_id
  GET  /api/stl/<id>   → serves the STL file for download/viewing
  GET  /api/health     → quick status check
"""

import os
import sys
import json
import uuid
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.vae import BoneVAE
from model.metrics import compute_scaffold_metrics
from geometry.mesh_export import voxel_to_stl, apply_cylinder_mask
from data.synthetic import generate_bone_sample

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

GENERATED_DIR = "generated"
FALLBACK_DIR = "fallbacks"
CHECKPOINT_PATH = "checkpoints/model_final.pth"
os.makedirs(GENERATED_DIR, exist_ok=True)

# ─── Load model at startup ────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BoneVAE(latent_dim=128).to(device)

model_loaded = False
if os.path.exists(CHECKPOINT_PATH):
    try:
        state = torch.load(CHECKPOINT_PATH, map_location=device)
        # Handle both raw state dict and checkpoint dict formats
        if "vae" in state:
            model.load_state_dict(state["vae"])
        else:
            model.load_state_dict(state)
        model.eval()
        model_loaded = True
        print(f"Model loaded from {CHECKPOINT_PATH} on {device}")
    except Exception as e:
        print(f"WARNING: Could not load model checkpoint: {e}")
        print("Serving fallback scaffolds only.")
else:
    print(f"WARNING: No checkpoint at {CHECKPOINT_PATH}")
    print("Train first: python model/train.py")
    print("Using fallback scaffolds for demo.")


def get_fallback(porosity_pct: int) -> tuple:
    """
    Serve a pre-generated scaffold closest to the requested porosity.
    Used when model isn't loaded or generation fails.
    """
    targets = [60, 65, 70, 75, 80]
    closest = min(targets, key=lambda t: abs(t - porosity_pct))

    stl_path = os.path.join(FALLBACK_DIR, f"scaffold_{closest}.stl")
    metrics_path = os.path.join(FALLBACK_DIR, f"metrics_{closest}.json")

    if not os.path.exists(stl_path):
        return None, None

    with open(metrics_path) as f:
        metrics = json.load(f)

    return stl_path, metrics


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "device": device,
        "fallbacks_available": os.path.exists(FALLBACK_DIR),
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    # porosity_pct: 60–80 (percentage, not fraction)
    porosity_pct = float(data.get("porosity_pct", 70))
    porosity_pct = max(55.0, min(85.0, porosity_pct))
    porosity_frac = porosity_pct / 100.0

    job_id = str(uuid.uuid4())[:8]
    stl_path = os.path.join(GENERATED_DIR, f"{job_id}.stl")

    if model_loaded:
        try:
            # 1. Generate voxel grid and shape it
            voxel = model.generate(porosity_frac, device=device)
            voxel = apply_cylinder_mask(voxel)

            # 2. Compute metrics
            metrics = compute_scaffold_metrics(voxel)

            # 3. Convert to STL
            voxel_to_stl(voxel, stl_path)

        except Exception as e:
            print(f"Generation failed: {e} — serving fallback")
            stl_path, metrics = get_fallback(int(porosity_pct))
            if stl_path is None:
                return jsonify({"error": "Generation failed and no fallback available."}), 500
            job_id = f"fallback_{int(porosity_pct)}"
    else:
        # No model — serve fallback
        stl_path, metrics = get_fallback(int(porosity_pct))
        if stl_path is None:
            return jsonify({
                "error": "Model not trained yet. Run: python model/train.py"
            }), 503
        job_id = f"fallback_{int(porosity_pct)}"

    return jsonify({
        "job_id": job_id,
        "metrics": metrics,
        "stl_url": f"/api/stl/{job_id}",
        "model_used": "trained_vae_gan" if model_loaded else "fallback",
    })


@app.route("/api/stl/<job_id>")
def serve_stl(job_id):
    # Check generated folder first
    gen_path = os.path.join(GENERATED_DIR, f"{job_id}.stl")
    if os.path.exists(gen_path):
        return send_file(
            gen_path,
            mimetype="model/stl",
            as_attachment=False,           # inline — lets Three.js fetch it
            download_name=f"scaffold_{job_id}.stl",
        )

    # Check fallback folder
    if job_id.startswith("fallback_"):
        pct = job_id.split("_")[1]
        fallback_path = os.path.join(FALLBACK_DIR, f"scaffold_{pct}.stl")
        if os.path.exists(fallback_path):
            return send_file(
                fallback_path,
                mimetype="model/stl",
                as_attachment=False,
                download_name=f"scaffold_{pct}pct.stl",
            )

    return jsonify({"error": "STL not found"}), 404

@app.route("/api/generate_batch", methods=["POST"])
def generate_batch():
    """Generate multiple scaffolds for comparison."""
    data = request.get_json(silent=True) or {}
    porosity_pct = float(data.get("porosity_pct", 70))
    porosity_pct = max(55.0, min(85.0, porosity_pct))
    porosity_frac = porosity_pct / 100.0
    count = min(int(data.get("count", 5)), 8)  # cap at 8

    results = []
    for i in range(count):
        job_id = str(uuid.uuid4())[:8]
        stl_path = os.path.join(GENERATED_DIR, f"{job_id}.stl")

        if model_loaded:
            try:
                voxel = model.generate(porosity_frac, device=device)
                voxel = apply_cylinder_mask(voxel)
                metrics = compute_scaffold_metrics(voxel)
                voxel_to_stl(voxel, stl_path)
            except Exception as e:
                stl_path, metrics = get_fallback(int(porosity_pct))
                if stl_path is None:
                    continue
                job_id = f"fallback_{int(porosity_pct)}"
        else:
            stl_path, metrics = get_fallback(int(porosity_pct))
            if stl_path is None:
                continue
            job_id = f"fallback_{int(porosity_pct)}"

        results.append({
            "job_id": job_id,
            "metrics": metrics,
            "stl_url": f"/api/stl/{job_id}",
            "index": i + 1,
        })

    return jsonify({"scaffolds": results, "count": len(results)})

@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run bone ingrowth simulation on a specific scaffold."""
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    steps = int(data.get("steps", 10))

    if not job_id:
        return jsonify({"error": "Missing job_id"}), 400

    # For hackathon demo: since we don't save the raw voxel grids to disk (only STLs),
    # we need to re-generate the voxel grid here. In production, we'd load it from a DB.
    # We use a fixed seed based on job_id so it generates the exact same scaffold.
    import hashlib
    seed = int(hashlib.md5(job_id.encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)

    # Hack to extract porosity from job_id if it's a fallback
    porosity_pct = 70
    if job_id.startswith("fallback_"):
        porosity_pct = int(job_id.split("_")[1])
    
    porosity_frac = porosity_pct / 100.0

    try:
        # Recreate the exact voxel grid
        if model_loaded:
            voxel = model.generate(porosity_frac, device=device)
        else:
            # Fallback for hackathon: if model not loaded, generate synthetic voxel grid
            # that looks like bone so simulation can still run
            np.random.seed(seed)
            voxel = generate_bone_sample(target_porosity=porosity_frac)
            
        # Make it a cylinder shape
        voxel = apply_cylinder_mask(voxel)
        
        # Run simulation
        import model.ingrowth as ingrowth
        results = ingrowth.simulate_ingrowth(voxel, n_steps=steps)
        
        # Save STL for each step showing bone fill
        step_urls = []
        for step_data in results:
            step_num = step_data["step"]
            sim_job_id = f"{job_id}_sim_{step_num}"
            stl_path = os.path.join(GENERATED_DIR, f"{sim_job_id}.stl")
            
            # Get combined grid (scaffold + new bone)
            combo_grid = ingrowth.get_ingrowth_at_step(voxel, step_num)
            
            # Export to STL (we treat both scaffold=1 and bone=2 as solid for the mesh)
            # but ideally the viewer would color them differently.
            # For now, we just export the filled mesh to show pores closing.
            binary_filled = (combo_grid > 0).astype(np.float32)
            voxel_to_stl(binary_filled, stl_path)
            
            step_data["stl_url"] = f"/api/stl/{sim_job_id}"
            step_urls.append(step_data)
            
        return jsonify({
            "job_id": job_id,
            "simulation_steps": step_urls
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Simulation failed: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting Synapse-Vis on http://localhost:5000")
    app.run(port=5000, debug=False)
