import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# Resolve all paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from graph_utils import get_projected_coords
SENSOR_COORDS = get_projected_coords(grid_size=64)

def interpolate_idw(sensor_values, grid_size=64, p=2.0):
    """
    sensor_values: [B, 10]
    Returns: [B, 1, 64, 64]
    """
    device = sensor_values.device
    B = sensor_values.shape[0]
    y, x = torch.meshgrid(torch.arange(grid_size, dtype=torch.float32, device=device), 
                          torch.arange(grid_size, dtype=torch.float32, device=device), indexing='ij')
    points = torch.stack([y.flatten(), x.flatten()], dim=1) # [64*64, 2]
    
    # distance: [10, 64*64]
    coords = SENSOR_COORDS.to(device)
    dist = torch.cdist(coords, points, p=2.0)
    
    # IDW weights
    dist = torch.clamp(dist, min=1e-3)
    weights = 1.0 / (dist ** p) # [10, 64*64]
    weights_sum = weights.sum(dim=0, keepdim=True) # [1, 64*64]
    weights = weights / weights_sum # [10, 64*64]
    
    # interpolate
    # sensor_values: [B, 10]
    # interpolated: [B, 64*64]
    interpolated = torch.matmul(sensor_values, weights)
    
    return interpolated.view(B, 1, grid_size, grid_size)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1) # [B, HW, C/8]
        k = self.key(x).view(B, -1, H * W) # [B, C/8, HW]
        v = self.value(x).view(B, -1, H * W) # [B, C, HW]
        
        attn = torch.softmax(torch.bmm(q, k), dim=-1) # [B, HW, HW]
        out = torch.bmm(v, attn.permute(0, 2, 1)) # [B, C, HW]
        out = out.view(B, C, H, W)
        return x + self.gamma * out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        # Input: 64x64 -> Downsample
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2) # 32x32
        
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2) # 16x16
        
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2) # 8x8
        
        self.bottleneck = nn.Sequential(
            DoubleConv(128, 256),
            SelfAttention(256),   # Self Attention injected here
            DoubleConv(256, 128)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 16x16
        self.conv_up1 = DoubleConv(128 + 128, 64)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 32x32
        self.conv_up2 = DoubleConv(64 + 64, 32)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 64x64
        self.conv_up3 = nn.Sequential(
            DoubleConv(32 + 32 + in_channels, 16),
            nn.Conv2d(16, out_channels, 1)
        )
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        
        bn = self.bottleneck(self.pool3(d3))
        
        u1 = self.conv_up1(torch.cat([self.up1(bn), d3], dim=1))
        u2 = self.conv_up2(torch.cat([self.up2(u1), d2], dim=1))
        u3 = self.conv_up3(torch.cat([self.up3(u2), d1, x], dim=1)) # Res connection to input
        
        return u3

class Discriminator(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        # 64x64 input
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2), # 32x32
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2), # 16x16
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2), # 8x8
            nn.Conv2d(128, 1, 8, stride=1, padding=0) # 1x1
        )
    def forward(self, condition, target):
        x = torch.cat([condition, target], dim=1)
        return self.model(x).view(-1, 1)

def compute_gradient_penalty(discriminator, condition, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(condition, interpolates)
    fake = torch.ones((real_samples.size(0), 1), device=device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class AQIDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        sensors = ['KE', 'K', 'VPW', 'D', 'C', 'BKC', 'CE', 'SN', 'CSIA', 'S']
        pm25_cols = [f"{s}_PM2.5" for s in sensors]
        
        # Using wind from KE node as approximate grid wind
        wind_speed = df['KE_Wind Speed'].values
        wind_dir = df['KE_Wind Direction'].values
        
        self.pm25_data = df[pm25_cols].values # [N, 10]
        # Dynamically compute and save/load max PM2.5 for correct normalization
        pm25_max_path = os.path.join(BASE_DIR, "pm25_max.npy")
        if not os.path.exists(pm25_max_path):
            pm25_max = np.max(self.pm25_data)
            np.save(pm25_max_path, pm25_max)
        else:
            pm25_max = np.load(pm25_max_path)
            
        self.pm25_data = self.pm25_data / pm25_max
        
        # Normalize Wind Speed to roughly [0, 1]
        self.wind_speed = wind_speed / 50.0 
        # Normalize Wind Direction to roughly [0, 1]
        self.wind_dir = wind_dir / 360.0 
        
    def __len__(self):
        return len(self.pm25_data) - 1
        
    def __getitem__(self, idx):
        input_pm25 = torch.tensor(self.pm25_data[idx], dtype=torch.float32)
        target_pm25 = torch.tensor(self.pm25_data[idx+1], dtype=torch.float32)
        
        ws = torch.tensor(self.wind_speed[idx], dtype=torch.float32)
        wd = torch.tensor(self.wind_dir[idx], dtype=torch.float32)
        
        return input_pm25, target_pm25, ws, wd

def get_condition_tensor(input_pm25, ws, wd, traffic_map):
    # input_pm25: [B, 10]
    heatmap = interpolate_idw(input_pm25, grid_size=64)
    
    B = heatmap.shape[0]
    
    ws_map = ws.view(B, 1, 1, 1).expand(B, 1, 64, 64)
    wd_map = wd.view(B, 1, 1, 1).expand(B, 1, 64, 64)
    
    t_map = traffic_map.expand(B, 1, 64, 64)
    
    condition = torch.cat([heatmap, t_map, ws_map, wd_map], dim=1)
    return condition

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = AQIDataset(os.path.join(BASE_DIR, 'mumbai_tensor_flattened.csv'))
    # Use smaller batch size so train loop processes faster
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    opt_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    # Persistent Static Traffic Map
    traffic_map_path = os.path.join(BASE_DIR, "traffic_map.npy")
    if not os.path.exists(traffic_map_path):
        torch.manual_seed(42)
        tm = torch.rand((1, 1, 64, 64))
        np.save(traffic_map_path, tm.numpy())
        traffic_map = tm.to(device)
    else:
        traffic_map = torch.tensor(np.load(traffic_map_path), dtype=torch.float32).to(device)
    
    lambda_gp = 10
    num_epochs = 1
    
    # Take a fraction of iterations for quick execution
    max_iters = 50 
    
    for epoch in range(num_epochs):
        for i, (in_p25, tgt_p25, ws, wd) in enumerate(dataloader):
            if i > max_iters:
                break
                
            in_p25 = in_p25.to(device)
            tgt_p25 = tgt_p25.to(device)
            ws = ws.to(device)
            wd = wd.to(device)
            
            condition = get_condition_tensor(in_p25, ws, wd, traffic_map)
            target_heatmap = interpolate_idw(tgt_p25, grid_size=64)
            
            # --- Train Critic (D) ---
            for _ in range(5):
                opt_D.zero_grad()
                
                real_validity = discriminator(condition, target_heatmap)
                
                fake_heatmap = generator(condition)
                fake_validity = discriminator(condition, fake_heatmap.detach())
                
                gp = compute_gradient_penalty(discriminator, condition, target_heatmap, fake_heatmap.detach(), device)
                
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
                d_loss.backward()
                opt_D.step()
                
            # --- Train Generator (G) ---
            opt_G.zero_grad()
            fake_heatmap = generator(condition)
            fake_validity = discriminator(condition, fake_heatmap)
            
            # L1 loss for image-to-image translation + WGAN loss
            l1_loss = F.l1_loss(fake_heatmap, target_heatmap)
            g_loss = -torch.mean(fake_validity) + 10.0 * l1_loss
            
            g_loss.backward()
            opt_G.step()
            
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    save_path = os.path.join(BASE_DIR, 'cgan_generator.pt')
    torch.save(generator.state_dict(), save_path)
    print(f"Training complete. Generator saved to {save_path}")

if __name__ == "__main__":
    train()
