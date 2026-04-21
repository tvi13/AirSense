import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import math
import time
import os

# Resolve all paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from graph_utils import get_spatial_graph

class HealerGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(HealerGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        return x

# --- Execution ---
if __name__ == '__main__':
    edge_index = get_spatial_graph(threshold=5.0)
    torch.save(edge_index, os.path.join(BASE_DIR, "edge_index.pt"))
    print(f"Constructed & Saved spatial graph. Found {edge_index.shape[1]} edges based on 5km threshold.")

    # 3. Data Loading
    print("Loading tensor...")
    tensor = np.load(os.path.join(BASE_DIR, "mumbai_tensor.npy"))
    # Standardize avoiding NaNs
    tensor_mean = np.nanmean(tensor, axis=(0, 1), keepdims=True)
    tensor_std = np.nanstd(tensor, axis=(0, 1), keepdims=True)
    tensor_std[tensor_std == 0] = 1.0 # avoid div by zero
    np.save(os.path.join(BASE_DIR, "norm_stats.npy"), {"mean": tensor_mean, "std": tensor_std})
    tensor = (tensor - tensor_mean) / tensor_std
    tensor = np.nan_to_num(tensor, nan=0.0)

    num_hours = tensor.shape[0]
    split = int(0.8 * num_hours)
    train_tensor = tensor[:split]
    test_tensor = tensor[split:]

    print(f"Total hours: {num_hours} (Train: {split}, Test: {num_hours - split})")

    train_dataset = [Data(x=torch.tensor(train_tensor[h], dtype=torch.float32), edge_index=edge_index) for h in range(len(train_tensor))]
    test_dataset = [Data(x=torch.tensor(test_tensor[h], dtype=torch.float32), edge_index=edge_index) for h in range(len(test_tensor))]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 4. Training
    model = HealerGAT(in_channels=7, hidden_channels=64, out_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    epochs = 30
    losses = []

    print("Training Healer GAT...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            batch_size_val = batch.ptr.size(0) - 1
            x_true = batch.x.clone()
            x_input = batch.x.clone()
            
            mask_idx = []
            for b in range(batch_size_val):
                start = batch.ptr[b].item()
                chosen = np.random.choice(10, 2, replace=False)
                mask_idx.extend((start + chosen).tolist())
            mask_idx = torch.tensor(mask_idx, dtype=torch.long)
            
            x_input[mask_idx, :4] = 0.0 # Mask PM2.5, PM10, NO2, CO
            out = model(x_input, batch.edge_index)
            
            loss = criterion(out[mask_idx], x_true[mask_idx, :4])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f}s.")

    # 5. Evaluation & Baselines
    model.eval()
    gnn_maelat = []
    mean_maelat = []
    reg_maelat = []

    # Simple Linear Regression Weights (Pre-calculated for demo, or we can fit a tiny one)
    # For this purpose, we'll use a Global Mean baseline and GAT predictions.
    global_mean = train_tensor[:, :, :4].mean(axis=(0, 1))

    print("Evaluating GAT vs Baselines (MAE/RMSE)...")
    with torch.no_grad():
        all_preds = []
        all_true = []
        all_mean = []
        
        for batch in test_loader:
            x_true = batch.x.clone()
            x_input = batch.x.clone()
            
            # Test 20% masking
            mask_idx = []
            for b in range(batch.ptr.size(0) - 1):
                start = batch.ptr[b].item()
                mask_idx.extend((start + np.random.choice(10, 2, replace=False)).tolist())
            mask_idx = torch.tensor(mask_idx, dtype=torch.long)
            
            x_input[mask_idx, :4] = 0.0
            preds = model(x_input, batch.edge_index)
            
            all_preds.append(preds[mask_idx])
            all_true.append(x_true[mask_idx, :4])
            all_mean.append(torch.tensor(global_mean, dtype=torch.float32).repeat(len(mask_idx), 1))

        full_preds = torch.cat(all_preds)
        full_true = torch.cat(all_true)
        full_mean = torch.cat(all_mean)
        
        gnn_mae = torch.mean(torch.abs(full_preds - full_true)).item()
        mean_mae = torch.mean(torch.abs(full_mean - full_true)).item()
        
        gnn_rmse = torch.sqrt(torch.mean((full_preds - full_true)**2)).item()
        mean_rmse = torch.sqrt(torch.mean((full_mean - full_true)**2)).item()

    print(f"\nRESULTS ON TEST SET:")
    print(f"{'Method':<15} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 40)
    print(f"{'Global Mean':<15} | {mean_mae:<10.4f} | {mean_rmse:<10.4f}")
    print(f"{'GAT (Proposed)':<15} | {gnn_mae:<10.4f} | {gnn_rmse:<10.4f}")
    print(f"Improvement: {((mean_mae-gnn_mae)/mean_mae)*100:.1f}% reduced error.")

    # Save Model
    save_path = os.path.join(BASE_DIR, "gnn_healer.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nHealer model (GAT) saved to {save_path}")
