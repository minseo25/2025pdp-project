import argparse
import os
import torch
from src.model_loader import load_model_strategy_a, load_model_strategy_b, load_model_baseline
from src.evaluate import measure_latency_throughput, calculate_perplexity, get_memory_usage

def main(args):
    # If distributed environment, get local rank
    is_distributed = 'LOCAL_RANK' in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if is_distributed and args.strategy == "int4_1gpu":
        if local_rank == 0:
            print("Error: int4_1gpu strategy cannot be run in distributed mode (torchrun).")
        return

    # Load model
    if args.strategy == "int4_1gpu":
        model, tokenizer = load_model_strategy_a(args.model_path)
    elif args.strategy == "int8_2gpu":
        model, tokenizer = load_model_strategy_b(args.model_path)
    elif args.strategy == "baseline":
        model, tokenizer = load_model_baseline(args.model_path)
    else:
        raise ValueError("Unknown strategy. Choose from 'int4_1gpu', 'int8_2gpu', 'baseline'.")

    # Only evaluate on rank 0
    if local_rank == 0:
        print("\n" + "="*50)
        print(f"Benchmark started: [Strategy: {args.strategy}] [Model: {args.model_path}]")
        print("="*50 + "\n")

        # 1. Measure performance (Latency & Throughput)
        print("--- Measure performance (Latency & Throughput) ---")
        prompt_text = "The art of parallel and distributed computing is"
        
        # Measure latency (batch_size=1)
        latency, _ = measure_latency_throughput(model, tokenizer, prompt_text, 128, 1)
        print(f"Latency (Batch Size 1): {latency:.4f} ms/token")

        # Measure throughput (batch_size=8)
        _, throughput = measure_latency_throughput(model, tokenizer, prompt_text, 128, 8)
        print(f"Throughput (Batch Size 8): {throughput:.4f} tokens/sec")

        # 2. Measure accuracy (Perplexity)
        print("\n--- Measure accuracy (Perplexity) ---")
        perplexity = calculate_perplexity(model, tokenizer)
        print(f"Perplexity (Wikitext-2): {perplexity:.4f}")

        # 3. Measure memory usage
        print("\n--- Measure memory usage ---")
        memory_usage = get_memory_usage()
        print(f"CPU RAM usage: {memory_usage['ram_gb']:.2f} GB")
        for gpu_mem in memory_usage['gpu_gb']:
            print(f"GPU {gpu_mem['gpu_id']} VRAM usage: {gpu_mem['allocated']:.2f} GB (Maximum: {gpu_mem['peak']:.2f} GB)")
        
        total_peak_vram = sum([gpu_mem['peak'] for gpu_mem in memory_usage['gpu_gb']])
        print(f"Total VRAM usage (Maximum): {total_peak_vram:.2f} GB")

        print("\n" + "="*50)
        print("Benchmark completed")
        print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM inference performance benchmark script")
    parser.add_argument("--model_path", type=str, required=True, help="Local model weight path (e.g., ./models/Llama-3-8b)")
    parser.add_argument("--strategy", type=str, required=True, choices=["int4_1gpu", "int8_2gpu", "baseline"], help="Inference strategy to run")
    
    args = parser.parse_args()
    main(args)