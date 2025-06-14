#!/bin/bash
# This script profiles the INT8 2-GPU strategy using nsys.
# Usage: ./profile_nsys.sh <model_path>

MODEL_PATH=$1
OUTPUT_PROFILE="profile_int8_2gpu"

echo "NVIDIA Nsight Systems profiling started..."
echo "Model path: ${MODEL_PATH}"
echo "Output file: ${OUTPUT_PROFILE}.nsys-rep"

nsys profile \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -o ${OUTPUT_PROFILE} \
    --force-overwrite=true \
    python profile_torch.py --model_path ${MODEL_PATH}

echo "Profiling completed. Open ${OUTPUT_PROFILE}.nsys-rep file in nsys-ui to view."