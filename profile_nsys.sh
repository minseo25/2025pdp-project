#!/bin/bash
# P2P Memory Transfer Profiling for device_map="auto" (Layer-wise Model Parallelism)
# Focus: cudaMemcpyPeerAsync and activation transfers between GPUs
# Usage: ./profile_nsys.sh <model_path>

MODEL_PATH=$1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results"
OUTPUT_PROFILE="${RESULTS_DIR}/${TIMESTAMP}_profile_p2p_layer_parallel"

echo "=== P2P Memory Transfer Profiler for Layer Parallelism ==="
echo "Model path: ${MODEL_PATH}"
echo "Results directory: ${RESULTS_DIR}"
echo "Output file: ${OUTPUT_PROFILE}.nsys-rep"
echo "Focus: Layer-wise model parallelism communication analysis"

# Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

# P2P memory transfer focused profiling (without GPU metrics due to permission requirements)
nsys profile \
    -t cuda,nvtx,cudnn,cublas \
    -o ${OUTPUT_PROFILE} \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-memory-usage=true \
    python profile_torch.py --model_path ${MODEL_PATH}

# Check if profiling was successful
if [ ! -f "${OUTPUT_PROFILE}.nsys-rep" ]; then
    echo "ERROR: Profiling failed. No .nsys-rep file was created."
    exit 1
fi

echo ""
echo "=== Generating P2P Transfer Analysis Reports ==="

# Generate P2P memory transfer statistics
nsys stats \
    --report cuda_api_sum \
    --format csv \
    -o ${RESULTS_DIR}/${TIMESTAMP}_p2p_transfers.csv \
    ${OUTPUT_PROFILE}.nsys-rep

# Generate GPU trace report
nsys stats \
    --report cuda_gpu_trace \
    --format csv \
    -o ${RESULTS_DIR}/${TIMESTAMP}_gpu_trace.csv \
    ${OUTPUT_PROFILE}.nsys-rep

echo "P2P communication profiling completed successfully."
echo ""
echo "=== ANALYSIS GUIDE FOR LAYER PARALLELISM ==="
echo "1. Main Profile: Open ${OUTPUT_PROFILE}.nsys-rep in nsys-ui"
echo "2. P2P Transfer Report: ${RESULTS_DIR}/${TIMESTAMP}_p2p_transfers.csv"
echo "   - Look for cudaMemcpyPeerAsync frequency and bandwidth"
echo "   - Identify activation transfer patterns between layers"
echo "3. GPU Trace Report: ${RESULTS_DIR}/${TIMESTAMP}_gpu_trace.csv"
echo "   - Memory usage patterns per GPU"
echo "   - Layer execution timeline"
echo ""
echo "Focus Areas in nsys-ui:"
echo "  ✓ CUDA API timeline: cudaMemcpyPeerAsync calls"
echo "  ✓ NVTX ranges: layer_forward_X timing comparison"
echo "  ✓ GPU utilization: idle time during P2P transfers"
echo "  ✓ Memory bandwidth: Check Timeline > GPU Utilization"
echo ""
echo "Communication overhead = (P2P transfer time / Total inference time)"
echo ""
echo "Note: GPU metrics collection disabled due to permission requirements on A100"