import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.model_loader import load_model_strategy_b

def main(args):
    # This script focuses on profiling the INT8 2-GPU strategy.
    # 'device_map="auto"' handles the multi-GPU distribution internally.
    model, tokenizer = load_model_strategy_b(args.model_path)
    
    prompt = "Profiling distributed systems requires careful instrumentation."
    
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

    print("Profiling results:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=20))
    
    # Save profiling results for visualization in TensorBoard
    trace_file = "trace_int8_2gpu.json"
    prof.export_chrome_trace(trace_file)
    print(f"Profiler trace saved to: {trace_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Profiler for multi-GPU strategy")
    parser.add_argument("--model_path", type=str, required=True, help="Local model weight path")
    args = parser.parse_args()
    main(args)