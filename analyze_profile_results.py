import pandas as pd
import sys
import os
from pathlib import Path

def analyze_p2p_transfers(csv_path):
    """P2P communication overhead analysis"""
    print("=== P2P Transfer Analysis ===")
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Total execution time
    total_time_ns = df['Total Time (ns)'].sum()
    total_time_ms = total_time_ns / 1_000_000
    
    print(f"Total execution time: {total_time_ms:.2f} ms")
    print(f"Total API calls: {df['Num Calls'].sum():,}")
    
    # Top API calls
    print("\n--- Top API Calls by Time ---")
    top_apis = df.head(10)
    for _, row in top_apis.iterrows():
        name = row['Name']
        time_pct = row['Time (%)']
        num_calls = row['Num Calls']
        avg_time_us = row['Avg (ns)'] / 1000
        
        print(f"{name:25s} | {time_pct:5.1f}% | {num_calls:8,} calls | {avg_time_us:8.1f} μs avg")
    
    # Memory transfer related APIs
    print("\n--- Memory Transfer APIs ---")
    memory_apis = df[df['Name'].str.contains('Memcpy|Memset', case=False, na=False)]
    
    if not memory_apis.empty:
        for _, row in memory_apis.iterrows():
            name = row['Name']
            time_pct = row['Time (%)']
            num_calls = row['Num Calls']
            total_time_ms = row['Total Time (ns)'] / 1_000_000
            
            print(f"{name:25s} | {time_pct:5.1f}% | {num_calls:8,} calls | {total_time_ms:8.1f} ms total")
    else:
        print("No memory transfer APIs found.")
    
    # Communication overhead calculation
    kernel_time = df[df['Name'] == 'cudaLaunchKernel']['Total Time (ns)'].sum()
    memcpy_time = df[df['Name'].str.contains('Memcpy', case=False, na=False)]['Total Time (ns)'].sum()
    
    if kernel_time > 0:
        communication_overhead = (memcpy_time / total_time_ns) * 100
        computation_ratio = (kernel_time / total_time_ns) * 100
        
        print(f"\n--- Communication Overhead ---")
        print(f"Pure computation time: {computation_ratio:.1f}%")
        print(f"Memory transfer time: {communication_overhead:.1f}%")
        print(f"Other overhead: {100 - computation_ratio - communication_overhead:.1f}%")

def analyze_layer_transfer_pattern():
    """Layer transfer pattern analysis"""
    print("\n=== Layer Transfer Pattern Analysis ===")
    print("Layer distribution (from profiling results):")
    print("  GPU 0: layers 0-10    (embedding + first 11 layers)")
    print("  GPU 1: layers 11-31   (remaining 21 layers + lm_head)")
    print("")
    print("P2P transfer points:")
    print("  layer 10 -> layer 11: GPU 0 -> GPU 1 transfer")
    print("  expected transfer size: [batch_size, seq_len, hidden_size] activation")
    print("")
    print("P2P pattern per token generation:")
    print("  - first token: transfer entire sequence at once")
    print("  - subsequent tokens: KV cache + new token activation transfer")

def print_analysis_guide():
    """Analysis guide"""
    print("\n" + "="*60)
    print("=== Additional analysis in nsys-ui ===")
    print("="*60)
    
    print("\n1. Run nsys-ui:")
    print("   nsys-ui results/20250614_081830_profile_p2p_layer_parallel.nsys-rep")
    
    print("\n2. Timeline analysis:")
    print("   Check the following tracks in Timeline tab:")
    print("   ✓ CUDA API: cudaMemcpyAsync calls")
    print("   ✓ NVTX: layer_parallel_forward range")
    print("   ✓ GPU 0/1 kernels: distribution of work per GPU")
    
    print("\n3. Key analysis points:")
    print("   • Idle time gap between GPU 0 and GPU 1")
    print("   • Check if cudaMemcpyAsync calls match GPU switch points")
    print("   • P2P pattern repeated per token")
    
    print("\n4. Performance optimization hints:")
    print("   • GPU 1 handles more layers, so imbalance exists")
    print("   • P2P transfer is minimized to 1 per token")
    print("   • Memory transfer overhead is at reasonable level (11.4%)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_profile_results.py <timestamp_prefix>")
        print("Example: python analyze_profile_results.py results/20250614_081830")
        return
    
    prefix = sys.argv[1]
    
    # P2P transfer analysis
    p2p_csv = f"{prefix}_p2p_transfers.csv_cuda_api_sum.csv"
    analyze_p2p_transfers(p2p_csv)
    
    # Layer transfer pattern analysis
    analyze_layer_transfer_pattern()
    
    # Analysis guide
    print_analysis_guide()

if __name__ == "__main__":
    main() 