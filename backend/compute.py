import platform
import psutil
import torch
import os

def get_compute_info():
    """
    Analyzes and returns information about the system's compute capabilities,
    including OS, CPU architecture, memory, and the optimal PyTorch device (GPU/CPU).
    """
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }
    
    # Memory details
    vm = psutil.virtual_memory()
    info["total_ram_gb"] = round(vm.total / (1024 ** 3), 2)
    info["available_ram_gb"] = round(vm.available / (1024 ** 3), 2)
    info["cpu_cores"] = psutil.cpu_count(logical=True)
    
    # PyTorch Compute Device
    optimal_device = "cpu"
    device_name = "CPU"
    has_gpu = False
    
    if torch.cuda.is_available():
        optimal_device = "cuda"
        has_gpu = True
        device_name = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        optimal_device = "mps"
        has_gpu = True
        device_name = "Apple Metal Performance Shaders (MPS)"
    
    info["optimal_device"] = optimal_device
    info["device_name"] = device_name
    info["has_gpu"] = has_gpu
    
    return info

def print_compute_summary():
    """Prints a human-readable summary of the compute capabilities."""
    info = get_compute_info()
    print("=== System Compute Summary ===")
    print(f"OS:           {info['os']} {info['os_release']} ({info['architecture']})")
    print(f"Python:       {info['python_version']}")
    print(f"CPU Cores:    {info['cpu_cores']}")
    print(f"Total RAM:    {info['total_ram_gb']} GB ({info['available_ram_gb']} GB available)")
    print("-" * 30)
    print(f"Optimal PyTorch Device: {info['optimal_device'].upper()}")
    print(f"Device Name:            {info['device_name']}")
    print(f"GPU Acceleration:       {'Enabled' if info['has_gpu'] else 'Disabled'}")
    print("==============================")

if __name__ == "__main__":
    print_compute_summary()
