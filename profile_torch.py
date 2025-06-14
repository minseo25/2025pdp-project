import argparse
import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity
from src.model_loader import load_model_strategy_b
import time

def analyze_layer_distribution(model):
    """레이어별 GPU 분포 분석 - P2P 전송 지점 예측"""
    print("\n=== Layer Distribution Analysis ===")
    if hasattr(model, 'hf_device_map'):
        layer_devices = {}
        for name, device in model.hf_device_map.items():
            layer_devices[name] = device
            print(f"  {name}: {device}")
        
        # P2P 전송 지점 예측
        print("\n--- P2P Transfer Points ---")
        devices = list(layer_devices.values())
        prev_device = None
        transfer_points = []
        
        for layer_name, device in layer_devices.items():
            if prev_device is not None and prev_device != device:
                transfer_point = f"{prev_device} -> {device}"
                transfer_points.append((layer_name, transfer_point))
                print(f"  At {layer_name}: {transfer_point}")
            prev_device = device
        
        print(f"  Total P2P transfer points: {len(transfer_points)}")
        return transfer_points
    else:
        print("  Device map not available")
        return []

def profile_layer_parallel_inference(model, tokenizer, inputs, run_idx):
    """레이어 병렬화의 P2P 전송 패턴 분석"""
    
    nvtx.range_push(f"inference_run_{run_idx}")
    
    # Pre-inference setup
    nvtx.range_push("pre_inference_setup")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Input을 첫 번째 레이어의 디바이스로 이동
    first_device = next(model.parameters()).device
    inputs_on_device = {k: v.to(first_device) for k, v in inputs.items()}
    nvtx.range_pop()
    
    with torch.no_grad():
        nvtx.range_push("input_preparation")
        # 입력 텐서 준비 완료
        nvtx.range_pop()
        
        # 토큰별 생성 과정에서 P2P 전송 추적
        nvtx.range_push("token_generation_loop")
        
        # 실제 모델 forward - 여기서 레이어 간 P2P 전송 발생
        nvtx.range_push("layer_parallel_forward")
        outputs = model.generate(
            **inputs_on_device,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            # 중간 상태를 유지해서 P2P 패턴 관찰
            output_hidden_states=False,  # 메모리 절약
            use_cache=True  # KV 캐시로 P2P 패턴 관찰
        )
        nvtx.range_pop()  # layer_parallel_forward
        
        nvtx.range_pop()  # token_generation_loop
        
        nvtx.range_push("output_collection")
        # 최종 출력을 CPU로 수집 - 마지막 GPU에서 CPU로 전송
        generated_ids = outputs.sequences
        if generated_ids.device != torch.device('cpu'):
            generated_ids_cpu = generated_ids.cpu()
        else:
            generated_ids_cpu = generated_ids
        nvtx.range_pop()
    
    # Post-inference timing
    nvtx.range_push("post_inference_sync")
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    nvtx.range_pop()
    
    nvtx.range_pop()  # inference_run_X
    
    return generated_ids_cpu, total_time

def analyze_gpu_memory_usage():
    """각 GPU의 메모리 사용량 분석"""
    print("\n=== GPU Memory Usage ===")
    for i in range(torch.cuda.device_count()):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB / {cached:.2f}GB / {total:.2f}GB")

def main(args):
    print("=== Layer Parallelism P2P Transfer Profiler ===")
    print("Target: cudaMemcpyPeerAsync and activation transfers")
    
    # Load model outside profiling scope
    print("\nLoading model (outside profiling scope)...")
    model, tokenizer = load_model_strategy_b(args.model_path)
    
    # 레이어 분포 및 P2P 전송 지점 분석
    transfer_points = analyze_layer_distribution(model)
    
    prompt = "Analyzing layer-wise model parallelism requires understanding activation transfer patterns between distributed GPU layers."
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print(f"\nInput length: {inputs['input_ids'].shape[1]} tokens")
    print(f"Expected P2P transfers per token: {len(transfer_points)}")
    
    # Warmup run (outside profiling)
    print("\nWarmup run (excluded from profiling)...")
    first_device = next(model.parameters()).device
    with torch.no_grad():
        _ = model.generate(
            **{k: v.to(first_device) for k, v in inputs.items()}, 
            max_new_tokens=3, 
            do_sample=False
        )
    
    # GPU memory analysis before profiling
    analyze_gpu_memory_usage()
    
    # Start CUDA profiler for nsys capture
    print("\nStarting CUDA profiler for P2P transfer analysis...")
    torch.cuda.profiler.start()
    
    # Multiple runs to analyze P2P transfer patterns
    num_runs = 3
    total_times = []
    
    for run_idx in range(num_runs):
        print(f"\nExecuting run {run_idx + 1}/{num_runs}...")
        
        generated_ids, run_time = profile_layer_parallel_inference(
            model, tokenizer, inputs, run_idx
        )
        
        total_times.append(run_time)
        
        # Output verification
        new_tokens = generated_ids.shape[1] - inputs['input_ids'].shape[1]
        expected_p2p_ops = new_tokens * len(transfer_points)
        
        print(f"Run {run_idx + 1}: {run_time:.3f}s, {new_tokens} tokens")
        print(f"Expected P2P operations: {expected_p2p_ops}")
    
    # Stop CUDA profiler
    torch.cuda.profiler.stop()
    
    # Performance summary
    avg_time = sum(total_times) / len(total_times)
    time_variation = max(total_times) - min(total_times)
    
    print(f"\n=== Performance Summary ===")
    print(f"Average inference time: {avg_time:.3f}s")
    print(f"Time variation: {time_variation:.3f}s")
    print(f"P2P transfer points: {len(transfer_points)}")
    
    print(f"\n=== P2P Analysis Guide ===")
    print("CSV Reports (auto-generated):")
    print("  • p2p_transfers.csv - cudaMemcpyPeerAsync statistics")
    print("  • gpu_trace.csv - GPU kernel execution timeline")
    print("")
    print("nsys-ui Timeline Analysis:")
    print("  1. CUDA API Track:")
    print("     - cudaMemcpyPeerAsync: activation transfers between layers")
    print("     - cudaMemcpy: fallback transfers (should be minimal)")
    print("  2. NVTX Ranges:")
    print("     - 'layer_parallel_forward': core computation with P2P")
    print("     - 'token_generation_loop': per-token P2P pattern")
    print("  3. GPU Utilization:")
    print("     - Idle periods = waiting for P2P transfers")
    print("     - Compare utilization between GPUs")
    print("  4. Memory Bandwidth:")
    print("     - PCIe/NVLink spikes during layer transitions")
    print("")
    print(f"P2P Efficiency = (Computation time) / (Total inference time)")
    print(f"Target: Minimize idle time between layer transitions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer Parallelism P2P Transfer Profiler")
    parser.add_argument("--model_path", type=str, required=True, help="Local model weight path")
    args = parser.parse_args()
    main(args)