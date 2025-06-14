import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.model_loader import load_model_strategy_b
from src.evaluate import measure_latency_throughput

def main(args):
    # This script focuses on profiling the INT8 2-GPU strategy.
    model, tokenizer = load_model_strategy_b(args.model_path)
    
    prompt = "Profiling distributed systems requires careful instrumentation."
    
    # Only run and output on rank 0
    if torch.distributed.get_rank() == 0:
        print("PyTorch Profiler for INT8 2-GPU strategy profiling started...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_inference"):
            # Target operations for profiling
            _ = model.generate(
                **tokenizer(prompt, return_tensors="pt").to(model.device),
                max_new_tokens=64
            )

    if torch.distributed.get_rank() == 0:
        print("Profiling results:")
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=20))
        # Save profiling results for visualization in TensorBoard
        prof.export_chrome_trace("trace.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/Llama-3-8b")
    args = parser.parse_args()
    main(args)