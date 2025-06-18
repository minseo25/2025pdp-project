import argparse
import os
import torch
import json
from datetime import datetime
from src.model_loader import load_model_strategy_a, load_model_strategy_b, load_model_baseline, load_model_strategy_b_tp
from src.evaluate import measure_latency_throughput, calculate_perplexity, get_memory_usage

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def main(args):
    # If distributed environment, get local rank
    is_distributed = 'LOCAL_RANK' in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if is_distributed and args.strategy == "int4_1gpu":
        if local_rank == 0:
            print("Error: int4_1gpu strategy cannot be run in distributed mode.")
        return

    # Load model
    if args.strategy == "int4_1gpu":
        model, tokenizer = load_model_strategy_a(args.model_path)
    elif args.strategy == "int8_2gpu":
        model, tokenizer = load_model_strategy_b(args.model_path, balanced_layers=True)
    elif args.strategy == "int8_2gpu_tp":
        model, tokenizer = load_model_strategy_b_tp(args.model_path)
    elif args.strategy == "baseline":
        model, tokenizer = load_model_baseline(args.model_path, balanced_layers=True)
    else:
        raise ValueError("Unknown strategy.")

    # Only evaluate on rank 0
    if local_rank == 0:
        print("\n" + "="*50)
        print(f"Benchmark started: [Strategy: {args.strategy}] [Model: {args.model_path}]")
        print("="*50 + "\n")

        # 1. Measure performance (Latency & Throughput)
        print("--- Measure performance (Latency & Throughput) ---")
        prompt_text = "The art of parallel and distributed computing is"
        
        latency, _ = measure_latency_throughput(model, tokenizer, prompt_text, 128, 1)
        print(f"Latency (Batch Size 1): {latency:.4f} ms/token")

        # Measure throughput (batch_size=8)
        _, throughput = measure_latency_throughput(model, tokenizer, prompt_text, 128, 8)
        print(f"Throughput (Batch Size 8): {throughput:.4f} tokens/sec")

        # 2. Measure accuracy (Perplexity)
        print("\n--- Measure accuracy (Perplexity) ---")
        try:
            perplexity = calculate_perplexity(model, tokenizer)
            print(f"Perplexity (Wikitext-2): {perplexity:.4f}")
        except Exception as e:
            print(f"Perplexity calculation failed: {e}")
            perplexity = -1.0

        # 3. Measure memory usage
        print("\n--- Measure memory usage ---")
        memory_usage = get_memory_usage()
        print(f"CPU RAM usage: {memory_usage['ram_gb']:.2f} GB")
        for gpu_mem in memory_usage['gpu_gb']:
            print(f"GPU {gpu_mem['gpu_id']} VRAM usage: {gpu_mem['allocated']:.2f} GB (Maximum: {gpu_mem['peak']:.2f} GB)")
        
        total_peak_vram = sum([gpu_mem['peak'] for gpu_mem in memory_usage['gpu_gb']])
        print(f"Total VRAM usage (Maximum): {total_peak_vram:.2f} GB")

        # 4. Save results to JSON file
        print("\n--- Saving results ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{args.strategy}.json"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        results = {
            "timestamp": timestamp,
            "model_path": args.model_path,
            "strategy": args.strategy,
            "latency_ms_per_token": round(latency, 4),
            "throughput_tokens_per_sec": round(throughput, 4),
            "perplexity": round(perplexity, 4) if perplexity > 0 else "CUDA_OOM_ERROR",
            "memory_usage": {
                "cpu_ram_gb": round(memory_usage['ram_gb'], 2),
                "gpu_memory_gb": memory_usage['gpu_gb'],
                "total_peak_vram_gb": round(total_peak_vram, 2)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {filepath}")

        print("\n" + "="*50)
        print("Benchmark completed")
        print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM inference performance benchmark script")
    parser.add_argument("--model_path", type=str, required=True, help="Local model weight path (e.g., ./models/Llama-3-8b)")
    # choices에 새로운 TP 전략을 추가합니다.
    parser.add_argument(
        "--strategy", 
        type=str, 
        required=True, 
        choices=["int4_1gpu", "int8_2gpu", "baseline", "int8_2gpu_tp"], 
        help="Inference strategy to run"
    )
    
    args = parser.parse_args()
    main(args)