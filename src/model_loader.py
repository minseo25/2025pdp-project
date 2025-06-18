import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import deepspeed
# import os

def load_model_strategy_a(model_id: str):
    """
    Strategy A: Load INT4 quantized model based on single GPU.
    - 4-bit NormalFloat (NF4) quantization
    - Double Quantization
    - Compute dtype: bfloat16 (utilizing A100 tensor cores)
    """
    print(f"Strategy A: INT4 quantization + single GPU model loading... ({model_id})")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    # Set pad token to eos token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda:0",
        local_files_only=True,
    )
    
    print("Model loading completed.")
    return model, tokenizer

def create_balanced_device_map_llama():
    """
    Create a balanced device map for Llama-3-8B (32 layers total)
    - GPU 0: layers 0-15 (16 layers)
    - GPU 1: layers 16-31 (16 layers)
    """
    device_map = {
        # Embedding layer
        "model.embed_tokens": 0,
        
        # First half of transformer layers (0-15) -> GPU 0
        **{f"model.layers.{i}": 0 for i in range(16)},
        
        # Second half of transformer layers (16-31) -> GPU 1
        **{f"model.layers.{i}": 1 for i in range(16, 32)},
        
        # Final layers
        "model.norm": 1,
        "model.rotary_emb": 1,
        "lm_head": 1,
    }
    
    return device_map

def create_balanced_device_map_qwen():
    """
    Create a balanced device map for Qwen2.5 (48 layers total)
    - GPU 0: layers 0-23 (24 layers)
    - GPU 1: layers 24-47 (24 layers)
    """
    device_map = {
        # Embedding layer
        "model.embed_tokens": 0,
        
        # First half of transformer layers (0-23) -> GPU 0
        **{f"model.layers.{i}": 0 for i in range(24)},
        
        # Second half of transformer layers (24-47) -> GPU 1
        **{f"model.layers.{i}": 1 for i in range(24, 48)},
        
        # Final layers
        "model.norm": 1,
        "model.rotary_emb": 1,
        "lm_head": 1,
    }
    
    return device_map

def load_model_strategy_b(model_id: str, balanced_layers: bool = True):
    """
    Strategy B: Load INT8 quantized model based on 2 GPUs.
    - 8-bit quantization
    - Balanced layer distribution (optional)
    
    Args:
        model_id: Model path
        balanced_layers: If True, use balanced 16:16 layer distribution
                        If False, use auto distribution
    """
    print(f"Strategy B (PP): INT8 quantization + 2-GPU pipeline parallelism model loading... ({model_id})")
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    # Set pad token to eos token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Choose device map based on balanced_layers flag
    if balanced_layers and "Llama-3-8b" in model_id:
        device_map = create_balanced_device_map_llama()
        print("Using balanced layer distribution:")
        print("  GPU 0: layers 0-15 (16 layers)")
        print("  GPU 1: layers 16-31 (16 layers)")
    elif balanced_layers and ("Qwen" in model_id or "qwen" in model_id):
        device_map = create_balanced_device_map_qwen()
        print("Using balanced layer distribution:")
        print("  GPU 0: layers 0-23 (24 layers)")
        print("  GPU 1: layers 24-47 (24 layers)")
    else:
        device_map = "auto"
        print("Using automatic layer distribution")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        local_files_only=True,
    )

    print("Model loading completed.")
    return model, tokenizer

def load_model_strategy_b_tp(model_id: str):
    """
    Strategy B with Tensor Parallelism: Load INT8 quantized model on 2 GPUs.
    
    TODO: INT8 + Tensor Parallelism implementation
    Current limitation: DeepSpeed CUDA accelerator does not support torch.int8 
    with tensor parallelism (tp_size > 1).
    
    Possible solutions:
    1. Use Triton kernels: init_inference(..., use_triton=True)
    2. Implement custom INT8 TP solution
    ...
    
    For now, this function is not implemented.
    """
    print(f"Strategy B (TP): INT8 + 2-GPU Tensor Parallelism - NOT IMPLEMENTED YET")
    raise NotImplementedError(
        "INT8 quantization with DeepSpeed Tensor Parallelism is not supported "
        "by CUDA accelerator. We will implement this in the future."
    )

def load_model_baseline(model_id: str, balanced_layers: bool = True):
    """
    Baseline: Load BFloat16 model based on 2 GPUs.
    - No quantization (BF16)
    - Balanced layer distribution (optional)
    
    Args:
        model_id: Model path
        balanced_layers: If True, use balanced 16:16 layer distribution
                        If False, use auto distribution
    """
    print(f"Baseline (PP): BF16 + 2-GPU pipeline parallelism model loading... ({model_id})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    # Set pad token to eos token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Choose device map based on balanced_layers flag
    if balanced_layers and "Llama-3-8b" in model_id:
        device_map = create_balanced_device_map_llama()
        print("Using balanced layer distribution:")
        print("  GPU 0: layers 0-15 (16 layers)")
        print("  GPU 1: layers 16-31 (16 layers)")
    elif balanced_layers and "Qwen2.5-14B" in model_id:
        device_map = create_balanced_device_map_qwen()
        print("Using balanced layer distribution:")
        print("  GPU 0: layers 0-23 (24 layers)")
        print("  GPU 1: layers 24-47 (24 layers)")
    else:
        device_map = "auto"
        print("Using automatic layer distribution")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        local_files_only=True,
    )

    print("Model loading completed.")
    return model, tokenizer