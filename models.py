import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# --- GNN Healer Architecture ---
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

# --- cGAN Components ---
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
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
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            DoubleConv(128, 256),
            SelfAttention(256),
            DoubleConv(256, 128)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(128 + 128, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(64 + 64, 32)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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
        u3 = self.conv_up3(torch.cat([self.up3(u2), d1, x], dim=1))
        return u3

# --- Spatial Utils (Lightweight) ---
from graph_utils import get_projected_coords
SENSOR_COORDS = get_projected_coords(grid_size=64)

def interpolate_idw(sensor_values, grid_size=64, p=2.0):
    device = sensor_values.device
    B = sensor_values.shape[0]
    y, x = torch.meshgrid(torch.arange(grid_size, dtype=torch.float32, device=device), 
                          torch.arange(grid_size, dtype=torch.float32, device=device), indexing='ij')
    points = torch.stack([y.flatten(), x.flatten()], dim=1)
    coords = SENSOR_COORDS.to(device)
    dist = torch.cdist(coords, points, p=2.0)
    dist = torch.clamp(dist, min=1e-3)
    weights = 1.0 / (dist ** p)
    weights_sum = weights.sum(dim=0, keepdim=True)
    weights = weights / weights_sum
    interpolated = torch.matmul(sensor_values, weights)
    return interpolated.view(B, 1, grid_size, grid_size)
