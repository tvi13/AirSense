import math
import torch
import numpy as np

# Spatial coordinates for 10 Mumbai sensors (GPS: Lat/Lon)
coords = {
    'KE': (19.0654, 72.8860),
    'K': (19.0728, 72.8795),
    'VPW': (19.1006, 72.8361),
    'D': (19.0178, 72.8478),
    'C': (18.9067, 72.8147),
    'BKC': (19.0590, 72.8656),
    'CE': (19.0522, 72.9001),
    'SN': (19.0402, 72.8616),
    'CSIA': (19.0896, 72.8656),
    'S': (19.0833, 72.8496)
}
nodes = ['KE', 'K', 'VPW', 'D', 'C', 'BKC', 'CE', 'SN', 'CSIA', 'S']

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth (specified in decimal degrees) return in km"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_spatial_graph(threshold=5.0):
    """Construct a PyTorch Geometric edge_index based on Haversine distance threshold."""
    edge_index_list = [[], []]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                dist = haversine(
                    coords[nodes[i]][0], coords[nodes[i]][1], 
                    coords[nodes[j]][0], coords[nodes[j]][1]
                )
                if dist <= threshold:
                    edge_index_list[0].append(i)
                    edge_index_list[1].append(j)
    return torch.tensor(edge_index_list, dtype=torch.long)

def get_projected_coords(grid_size=64):
    """
    Min-Max scale actual GPS coordinates to fit into a synthetic grid_size * grid_size matrix 
    Returns: a torch.tensor of shape [num_nodes, 2] containing the [y, x] grid coordinates
    """
    lats = [coords[n][0] for n in nodes]
    lons = [coords[n][1] for n in nodes]
    
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Simple linear mapping:
    # 0 -> min_lat/lon, (grid_size-1) -> max_lat/lon
    # We swap to [y, x] meaning [lat, lon]
    projected = []
    for n in nodes:
        y = (coords[n][0] - min_lat) / (max_lat - min_lat) * (grid_size - 1)
        x = (coords[n][1] - min_lon) / (max_lon - min_lon) * (grid_size - 1)
        projected.append([y, x])
        
    return torch.tensor(projected, dtype=torch.float32)
