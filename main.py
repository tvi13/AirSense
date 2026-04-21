from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import uvicorn
import io
import base64
from PIL import Image

# Resolve all paths relative to this file's directory, not the cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

class PolicyRequest(BaseModel):
    traffic_modifier: float = 100.0
    industrial_modifier: float = 100.0
    pm25: float = 12.0
    pm10: float = 18.0
    no2: float = 22.0
    o3: float = 15.0
    co: float = 0.6
    wind_speed: float = 12.0
    wind_direction: float = 0.0
    traffic_density: float = 0.5 # 0.0 to 1.0 baseline
    sensor_values: Optional[List[float]] = None

class ImputeRequest(BaseModel):
    sensor_values: List[float] # PM2.5 of all 10 nodes (some might be 0/crashed)
    pm10: float = 18.0
    no2: float = 22.0
    co: float = 0.6
    temperature: float = 28.0
    wind_speed: float = 12.0
    wind_direction: float = 0.0
    node_id: int # The crashed node to heal

# --- Dataset Statistics for Normalization ---
# Loaded dynamically to match training stats exactly
import numpy as np
_stats = np.load(os.path.join(BASE_DIR, "norm_stats.npy"), allow_pickle=True).item()
DATA_MEAN = torch.tensor(_stats["mean"].flatten(), dtype=torch.float32)
DATA_STD = torch.tensor(_stats["std"].flatten(), dtype=torch.float32)

# --- Model Classes & Tools ---
from models import HealerGAT, Generator, interpolate_idw
import gc

# Load persistence dependencies needed for inference
EDGE_INDEX = torch.load(os.path.join(BASE_DIR, "edge_index.pt"), map_location=torch.device('cpu'), weights_only=True)
try:
    TRAFFIC_BASE = torch.tensor(np.load(os.path.join(BASE_DIR, "traffic_map.npy")), dtype=torch.float32)
    PM25_MAX = float(np.load(os.path.join(BASE_DIR, "pm25_max.npy")))
except:
    TRAFFIC_BASE = torch.rand((1, 1, 64, 64))
    PM25_MAX = 200.0

# --- Load Models ---
try:
    healer_model = HealerGAT(in_channels=7, hidden_channels=64, out_channels=4)
    healer_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "gnn_healer.pt"), map_location=torch.device('cpu'), weights_only=True))
    healer_model.eval()
    print("gnn_healer.pt loaded successfully.")
except Exception as e:
    print(f"Error loading gnn_healer.pt: {e}")

try:
    cgan_model = Generator(in_channels=4, out_channels=1)
    cgan_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "cgan_generator.pt"), map_location=torch.device('cpu'), weights_only=True))
    cgan_model.eval()
    print("cgan_generator.pt loaded successfully.")
except Exception as e:
    print(f"Error loading cgan_generator.pt: {e}")

@app.get("/health")
def health_check():
    return {"status": "Models loaded successfully!"}

@app.post("/api/impute")
def impute_data(req: ImputeRequest):
    # 1. Construct node feature matrix (10 nodes, 7 features)
    # Feature Order: [PM2.5, PM10, NO2, CO, Temp, WindSpeed, WindDir]
    x = torch.zeros((10, 7))
    
    for i in range(10):
        # Fill sensor-specific PM2.5 (might be 0 for the crashed node)
        x[i, 0] = req.sensor_values[i] if i < len(req.sensor_values) else 0.0
        # Fill global/approximate other pollutants and weather for all nodes
        x[i, 1] = req.pm10
        x[i, 2] = req.no2
        x[i, 3] = req.co
        x[i, 4] = req.temperature
        x[i, 5] = req.wind_speed
        x[i, 6] = req.wind_direction

    # 2. Standardize Input
    x_norm = (x - DATA_MEAN) / DATA_STD
    
    # Ensure the crashed node's primary features are zeroed out for the GNN to "fill"
    crashed_node = req.node_id if 0 <= req.node_id < 10 else 0
    x_norm[crashed_node, :4] = 0.0 
    
    # 3. Use precomputed spatial edges
    edge_index = EDGE_INDEX
    
    # 4. Inference
    with torch.no_grad():
        out_norm = healer_model(x_norm, edge_index)
        
    # Derive true model confidence by seeing how perfectly it reconstructs known safe neighbor nodes
    healthy_mask = torch.ones(10, dtype=torch.bool)
    healthy_mask[crashed_node] = False
    healthy_input = x_norm[healthy_mask, 0]
    healthy_output = out_norm[healthy_mask, 0]
    mae = torch.nn.functional.l1_loss(healthy_output, healthy_input).item()
    # Confidence is mapped: MAE 0 -> 99%, High MAE -> Dropoff
    calc_conf = max(0.01, min(0.99, 1.0 / (1.0 + mae * 1.5)))
    
    # 5. De-standardize Output (Restore actual PM2.5 unit)
    # out_norm has shape [10, 4] corresponding to [PM2.5, PM10, NO2, CO]
    pm25_norm = out_norm[crashed_node, 0].item()
    healed_pm25 = (pm25_norm * DATA_STD[0].item()) + DATA_MEAN[0].item()
    
    # Safety clamp
    healed_pm25 = max(1.0, healed_pm25)
    
    return {
        "status": "success", 
        "crashed_node": crashed_node, 
        "healed_pm25": float(healed_pm25),
        "confidence": round(calc_conf, 2)
    }

@app.post("/api/simulate")
def simulate(request: PolicyRequest):
    import numpy as np
    
    # 1. PM2.5 Channel (Heatmap)
    if request.sensor_values and len(request.sensor_values) == 10:
        s_vals = torch.tensor([request.sensor_values], dtype=torch.float32) / PM25_MAX
        heatmap = interpolate_idw(s_vals, grid_size=64)
    else:
        # Fallback to single hotspot if sensor data is missing
        heatmap = torch.zeros(1, 1, 64, 64)
        base_pm = (request.pm25 / PM25_MAX)
        y, x = torch.meshgrid(torch.arange(64, dtype=torch.float32), torch.arange(64, dtype=torch.float32), indexing='ij')
        dist = torch.sqrt((x - 32)**2 + (y - 32)**2)
        heatmap[0, 0] = base_pm * torch.exp(-dist / 15.0)
    
    # Apply industrial modifier to the heatmap
    heatmap = heatmap * (request.industrial_modifier / 100.0)

    # 2. Traffic Channel (Apply modifier to persistent reproducible traffic map)
    t_map = TRAFFIC_BASE * (request.traffic_modifier / 100.0)
    
    # 3. Wind Speed Channel (Normalized 0-50 to 0-1)
    ws_map = torch.full((1, 1, 64, 64), request.wind_speed / 50.0)
    
    # 4. Wind Direction Channel (Normalized 0-360 to 0-1)
    wd_map = torch.full((1, 1, 64, 64), request.wind_direction / 360.0)
    
    # Build 4-channel condition tensor
    condition = torch.cat([heatmap, t_map, ws_map, wd_map], dim=1)
    
    with torch.no_grad():
        output = cgan_model(condition)
    
    # Convert to scalar array 0.0 - 1.0
    output_array = output.squeeze().cpu().numpy()
    # Normalize for display - more robust version
    out_min, out_max = output_array.min(), output_array.max()
    if out_max - out_min > 0.1:
        output_array = (output_array - out_min) / (out_max - out_min)
        
    # Colormap transformation: 0=Green, 0.25=Yellow, 0.5=Orange, 0.75=Red, 1.0=Purple
    H, W = output_array.shape
    rgba_img = np.zeros((H, W, 4), dtype=np.uint8)
    
    # IQAir standard colors
    def get_color(v):
        if v < 0.2: # Good
            t = v / 0.2
            return [80 + 175 * t, 220, 100 - 50 * t]
        elif v < 0.4: # Moderate
            t = (v - 0.2) / 0.2
            return [255, 220 - 80 * t, 50 - 20 * t]
        elif v < 0.6: # Unhealthy for Sensitive
            t = (v - 0.4) / 0.2
            return [255, 140 - 90 * t, 30 + 20 * t]
        elif v < 0.8: # Unhealthy
            t = (v - 0.6) / 0.2
            return [220 - 80 * t, 50, 50 + 110 * t]
        else: # Hazardous
            t = (v - 0.8) / 0.2
            return [140 - 40 * t, 50 - 30 * t, 160 - 110 * t]

    for y_idx in range(H):
        for x_idx in range(W):
            v = output_array[y_idx, x_idx]
            r, g, b = get_color(v)
                
            # Transparent clean air -> Opaque dirty clusters
            alpha = max(0, min(200, int(v * 240)))
            rgba_img[y_idx, x_idx] = [int(r), int(g), int(b), alpha]
            
    img = Image.fromarray(rgba_img, mode="RGBA")
    from PIL import ImageFilter
    # Increase blur radius to strongly smooth out the 64x64 cGAN grid
    img = img.filter(ImageFilter.GaussianBlur(radius=2.5)) 
    # Upscale significantly to prevent any pixelation in the frontend
    img = img.resize((512, 512), resample=Image.LANCZOS)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # Explicit memory cleanup for Render stability
    del img, output, rgba_img
    gc.collect()
    
    return {"status": "success", "image_b64": b64_str}


class SummaryRequest(BaseModel):
    location: str
    pm25: float
    pm10: float
    no2: float
    o3: float
    co: float
    wind: float

@app.post("/api/summary")
def get_ai_summary(req: SummaryRequest):
    import requests
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        return {"status": "error", "summary": "GROQ_API_KEY environment variable not set.", "recommendation": "Set the GROQ_API_KEY env var and restart the server."}
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
    Analyze the air quality in {req.location} with these stats:
    PM2.5: {req.pm25}, PM10: {req.pm10}, NO2: {req.no2}, O3: {req.o3}, CO: {req.co}, Wind: {req.wind} km/h.
    
    Provide:
    1. A 1-2 sentence professional summary of the situation.
    2. A specific, actionable recommendation for city officials or residents (e.g. 'Reduce heavy traffic', 'Construction pause').
    
    Format as:
    SUMMARY: [text]
    RECOMMENDATION: [text]
    """
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are an expert environmental dashboard AI. Provide only the requested SUMMARY and RECOMMENDATION in the exact format specified. No intro, no conversational filler."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        summary = "No summary available."
        recommendation = "No recommendation available."
        
        if "SUMMARY:" in content and "RECOMMENDATION:" in content:
            parts = content.split("RECOMMENDATION:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            recommendation = parts[1].strip()
        else:
            summary = content
            
        return {"status": "success", "summary": summary, "recommendation": recommendation}
    except Exception as e:
        print(f"Groq API Error: {e}")
        return {"status": "error", "summary": "Insights currently unavailable due to network delay.", "recommendation": "Please try again shortly."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
