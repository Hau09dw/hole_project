# scripts/benchmark.py
import os, time, yaml, torch
from models.yolov12_bifpn import YOLOv12_BiFPN

def try_thop(model, img_size):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, img_size, img_size).to(next(model.parameters()).device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        # thop trả FLOPs cho toàn forward; chuyển về GFLOPs, Params (M)
        return flops/1e9, params/1e6
    except Exception as e:
        return None, None

@torch.no_grad()
def measure_latency(model, device, img_size=640, warmup=10, iters=100):
    model.eval().to(device)
    x = torch.randn(1, 3, img_size, img_size, device=device)
    # warmup
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # measure
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    total = t1 - t0
    lat_ms = (total / iters) * 1000.0
    thr = iters / total
    return lat_ms, thr

def main():
    with open("configs/configs.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    img_size = cfg["data"]["img_size"]

    # build model
    device_gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = YOLOv12_BiFPN(num_classes=cfg["data"]["num_classes"]).to(device_gpu)

    # Params (count)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # M

    # FLOPs (optional, requires thop)
    gflops, thop_params_m = try_thop(model, img_size)

    # Latency/Throughput GPU
    lat_gpu_ms, thr_gpu = measure_latency(model, device_gpu, img_size)
    # Latency/Throughput CPU
    device_cpu = torch.device("cpu")
    model_cpu = YOLOv12_BiFPN(num_classes=cfg["data"]["num_classes"]).to(device_cpu)
    lat_cpu_ms, thr_cpu = measure_latency(model_cpu, device_cpu, img_size)

    print("\n=== MODEL COST & RUNTIME ===")
    print(f"Params (count): {total_params:.2f} M")
    if gflops is not None:
        print(f"GFLOPs (thop): {gflops:.2f} G, Params(thop): {thop_params_m:.2f} M")
    else:
        print("GFLOPs (thop): N/A (install thop for FLOPs)")

    print(f"GPU Latency (bs=1): {lat_gpu_ms:.2f} ms  | Throughput: {thr_gpu:.2f} img/s")
    print(f"CPU Latency (bs=1): {lat_cpu_ms:.2f} ms  | Throughput: {thr_cpu:.2f} img/s")

if __name__ == "__main__":
    main()
