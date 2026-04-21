import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Resolve all paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use Agg backend for headless plotting
plt.switch_backend('Agg')

# --- 1. GNN Performance Comparison (MAE/RMSE) ---
labels = ['MAE', 'RMSE']
mean_vals = [0.0, 0.0]
gnn_vals = [0.0, 0.0]

try:
    with open(os.path.join(BASE_DIR, "metrics_report.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("MEAN_MAE:"): mean_vals[0] = float(line.split(":")[1].strip())
            elif line.startswith("MEAN_RMSE:"): mean_vals[1] = float(line.split(":")[1].strip())
            elif line.startswith("GNN_MAE:"): gnn_vals[0] = float(line.split(":")[1].strip())
            elif line.startswith("GNN_RMSE:"): gnn_vals[1] = float(line.split(":")[1].strip())
except FileNotFoundError:
    print("metrics_report.txt not found. Please run evaluate_models.py first.")

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, mean_vals, width, label='Global Mean (Baseline)', color='#e74c3c')
rects2 = ax.bar(x + width/2, gnn_vals, width, label='GNN Healer (Proposed)', color='#2ecc71')

ax.set_ylabel('Normalized Error Value')
ax.set_title('GNN Resilience Performance Analysis', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(BASE_DIR, "gnn_performance.png"), dpi=300, bbox_inches='tight')
print("Saved gnn_performance.png")

# --- 2. GNN Imputation Time-Series Demo ---
from train_healer import HealerGAT
from graph_utils import get_spatial_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
healer_model = HealerGAT(in_channels=7, hidden_channels=64, out_channels=4).to(device)
healer_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "gnn_healer.pt"), map_location=device, weights_only=True))
healer_model.eval()

# Precomputed edges on device
edge_index = get_spatial_graph(threshold=5.0).to(device)

print("Running actual GNN inference on held-out test block for visualization...")
tensor = np.load(os.path.join(BASE_DIR, "mumbai_tensor.npy"))
_stats = np.load(os.path.join(BASE_DIR, "norm_stats.npy"), allow_pickle=True).item()
tensor_norm = (tensor - _stats["mean"]) / _stats["std"]
tensor_norm = np.nan_to_num(tensor_norm, nan=0.0)

time_steps = 48
start_idx = int(0.8 * tensor_norm.shape[0]) + 100 # Pick a block from test set
target_node = 3 # E.g. node 3 fails

real_values = []
healed_values = []
mean_baseline = []

for t_idx in range(start_idx, start_idx + time_steps):
    x_true = torch.tensor(tensor_norm[t_idx], dtype=torch.float32).to(device)
    x_input = x_true.clone()
    
    # Simulate hardware crash across primary pollutants on target node
    x_input[target_node, :4] = 0.0 
    
    with torch.no_grad():
        out = healer_model(x_input, edge_index)
        
    real_values.append(x_true[target_node, 0].item()) # Actual normalized PM2.5
    healed_values.append(out[target_node, 0].item()) # Model prediction
    mean_baseline.append(0.0) # Normalized datasets naturally have a 0.0 mean

plt.figure(figsize=(12, 5))
plt.plot(range(time_steps), real_values, 'k--', label='Original Stream', alpha=0.5)
plt.plot(range(time_steps), mean_baseline, 'r:', label='Baseline (Global Mean)')
plt.plot(range(time_steps), healed_values, 'g-', label='GNN Reconstruction (Spatial Context)', linewidth=2)
plt.fill_between(range(time_steps), mean_baseline, healed_values, color='green', alpha=0.1)
plt.title("Real-time Sensor Failure Reconstruction Trace (Authentic Inference)", fontsize=13)
plt.ylabel("Normalized Pollutant Value")
plt.xlabel("Time (Hours)")
plt.legend()
plt.savefig(os.path.join(BASE_DIR, "gnn_imputation_demo.png"), dpi=300, bbox_inches='tight')
print("Saved gnn_imputation_demo.png")

# --- 3. cGAN vs. IDW Heatmap Comparison ---
from simulator import interpolate_idw, Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
cgan_model = Generator(in_channels=4, out_channels=1).to(device)
cgan_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "cgan_generator.pt"), map_location=device, weights_only=True))
cgan_model.eval()

traffic_map = torch.tensor(np.load(os.path.join(BASE_DIR, "traffic_map.npy")), dtype=torch.float32).to(device)

pm25_vals = torch.tensor([[0.2, 0.8, 0.4, 0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.6]], dtype=torch.float32).to(device)
idw_map_tensor = interpolate_idw(pm25_vals, grid_size=64)

# Create neutral wind conditions for baseline comparison
ws_map_neutral = torch.full((1, 1, 64, 64), 0.5, device=device)
wd_map_neutral = torch.full((1, 1, 64, 64), 0.5, device=device)

# Run actual cGAN prediction
condition_base = torch.cat([idw_map_tensor, traffic_map, ws_map_neutral, wd_map_neutral], dim=1)
with torch.no_grad():
    cgan_refined_tensor = cgan_model(condition_base)

idw_map = idw_map_tensor.squeeze().cpu().numpy()
cgan_refined = cgan_refined_tensor.squeeze().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im1 = axes[0].imshow(idw_map, cmap='magma')
axes[0].set_title("Baseline: Spatial Interpolation (IDW)", fontsize=12)
axes[0].axis('off')
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(cgan_refined, cmap='magma')
axes[1].set_title("Proposed: Physics-Aware Generation (cGAN)", fontsize=12)
axes[1].axis('off')
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.suptitle("Heatmap Fidelity: Baseline vs. Actual Model Generative Refinement", fontsize=15, fontweight='bold')
plt.savefig(os.path.join(BASE_DIR, "cgan_comparison.png"), dpi=300, bbox_inches='tight')
print("Saved cgan_comparison.png")

# --- 4. Wind Physics Demonstration ---
# Pass actual directional physics through the real GAN
def get_wind_map_generative(direction_deg):
    direction_norm = min(1.0, max(0.0, direction_deg / 360.0))
    wd_map = torch.full((1, 1, 64, 64), direction_norm, dtype=torch.float32, device=device)
    ws_map = torch.full((1, 1, 64, 64), 0.8, dtype=torch.float32, device=device) 
    
    cond = torch.cat([idw_map_tensor, traffic_map, ws_map, wd_map], dim=1)
    with torch.no_grad():
        out = cgan_model(cond)
    return out.squeeze().cpu().numpy()

wind_n = get_wind_map_generative(90) 
wind_s = get_wind_map_generative(270)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(wind_n, cmap='magma')
axes[0].set_title("Physics Check: North Wind Drift")
axes[0].axis('off')
axes[1].imshow(wind_s, cmap='magma')
axes[1].set_title("Physics Check: South Wind Drift")
axes[1].axis('off')
plt.savefig(os.path.join(BASE_DIR, "wind_physics.png"), dpi=300, bbox_inches='tight')
print("Saved wind_physics.png")

print("\nAll technical report visuals generated successfully.")
