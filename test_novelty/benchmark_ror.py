import torch
import time
import numpy as np
import sys
import rospkg

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
sys.path.insert(0, pkg_path)

from vision_processing import fast_perception_module

def benchmark(num_points=5000, radius=0.05, min_neighbors=5, iterations=100):
    print(f"========== BENCHMARK RADIUS OUTLIER REMOVAL ==========")
    print(f"Points: {num_points} | Radius: {radius} | Min Neighbors: {min_neighbors}")
    print(f"Itérations: {iterations}\n")

    # Générer un nuage de points aléatoires sur GPU
    device = torch.device('cuda')
    pts_gpu = torch.rand((num_points, 3), device=device) * 2.0 - 1.0
    
    # Préchauffage (Warmup)
    for _ in range(5):
        # GPU
        dists = torch.cdist(pts_gpu, pts_gpu)
        mask_ror = (dists < radius).sum(dim=1) >= min_neighbors
        pts_gpu[mask_ror]
        # C++
        pts_np = pts_gpu.cpu().numpy().astype(np.float32)
        fast_perception_module.radius_outlier_removal(pts_np, radius, min_neighbors)
    
    torch.cuda.synchronize()
    
    # ---------------------------------------------------------
    # 1. TEST FULL GPU (PyTorch cdist)
    # ---------------------------------------------------------
    start_gpu = time.perf_counter()
    for _ in range(iterations):
        dists = torch.cdist(pts_gpu, pts_gpu)
        mask_ror = (dists < radius).sum(dim=1) >= min_neighbors
        pts_filtered_gpu = pts_gpu[mask_ror]
        torch.cuda.synchronize()
    end_gpu = time.perf_counter()
    
    time_gpu_ms = ((end_gpu - start_gpu) / iterations) * 1000.0
    
    # ---------------------------------------------------------
    # 2. TEST HYBRIDE (GPU -> CPU C++ -> GPU)
    # ---------------------------------------------------------
    start_cpp = time.perf_counter()
    for _ in range(iterations):
        # Simulation exacte de la boucle de ton noeud (incluant les transferts de mémoire !)
        pts_np = pts_gpu.cpu().numpy().astype(np.float32)
        pts_filtered_np = fast_perception_module.radius_outlier_removal(pts_np, radius, min_neighbors)
        pts_filtered_tensor = torch.tensor(pts_filtered_np, device=device)
        torch.cuda.synchronize()
    end_cpp = time.perf_counter()
    
    time_cpp_ms = ((end_cpp - start_cpp) / iterations) * 1000.0
    
    print(f"🚀 PyTorch (Full GPU) : {time_gpu_ms:.2f} ms par itération")
    print(f"⚡ Pybind11 PCL (C++) : {time_cpp_ms:.2f} ms par itération (transferts inclus)")
    
    if time_cpp_ms < time_gpu_ms:
        print(f"\n=> Le C++ est {time_gpu_ms / time_cpp_ms:.1f}x plus rapide !")
    else:
        print(f"\n=> Le GPU est {time_cpp_ms / time_gpu_ms:.1f}x plus rapide !")
        
    print("======================================================")

if __name__ == "__main__":
    benchmark(num_points=5000)
    benchmark(num_points=15000)
