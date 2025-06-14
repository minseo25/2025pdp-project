import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

def load_model_strategy_b(model_id: str):
    """
    Strategy B: Load INT8 quantized model based on 2 GPUs.
    - 8-bit quantization
    - Tensor Parallelism enabled (tp_plan="auto")
    """
    print(f"Strategy B: INT8 quantization + 2-GPU tensor parallelism model loading... ({model_id})")
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    # Set pad token to eos token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    print("Model loading completed.")
    return model, tokenizer

def load_model_baseline(model_id: str):
    """
    Baseline: Load BFloat16 model based on 2 GPUs.
    - No quantization (BF16)
    - Tensor Parallelism enabled (tp_plan="auto")
    """
    print(f"Baseline: BF16 + 2-GPU tensor parallelism model loading... ({model_id})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    # Set pad token to eos token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    print("Model loading completed.")
    return model, tokenizer