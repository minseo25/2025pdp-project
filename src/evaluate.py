import torch
import time
import psutil
from datasets import load_dataset
from tqdm import tqdm

def get_memory_usage():
    """Return current system and GPU memory usage in GB."""
    # CPU memory
    ram_usage = psutil.Process().memory_info().rss / (1024 ** 3)
    
    # GPU memory
    gpu_usage = []
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        gpu_usage.append({
            "gpu_id": i,
            "allocated": torch.cuda.memory_allocated(i) / (1024 ** 3),
            "peak": torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        })
    
    return {"ram_gb": ram_usage, "gpu_gb": gpu_usage}

def measure_latency_throughput(model, tokenizer, prompt: str, num_tokens_to_generate: int, batch_size: int):
    """Measure latency and throughput."""
    prompts = [prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # warmup
    _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    outputs = model.generate(**inputs, max_new_tokens=num_tokens_to_generate, do_sample=False)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_tokens = batch_size * num_tokens_to_generate
    
    latency_per_token = (total_time / total_tokens) * 1000  # ms/token
    throughput = total_tokens / total_time  # tokens/sec

    return latency_per_token, throughput

def calculate_perplexity(model, tokenizer):
    """
    Calculate Perplexity for Wikitext-2 dataset manually.
    """
    print("Calculating Perplexity manually...")
    
    # 1. Load test dataset and use half of the data
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    half_size = len(test["text"]) // 2
    sample_texts = test["text"][:half_size]  # 절반만 사용
    encodings = tokenizer("\n\n".join(sample_texts), return_tensors="pt")

    # 2. Setup for Perplexity calculation
    max_length = model.config.max_position_embeddings
    stride = 512  # stride for long sequences
    seq_len = encodings.input_ids.size(1)

    nlls = []  # Negative Log-Likelihoods
    prev_end_loc = 0
    
    # Use tqdm to show progress
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Ignore previous context for loss calculation

        if input_ids.size(1) == 0:
            continue
            
        # 3. Calculate Loss (disable gradient calculation)
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # Loss is the average value, so multiply by the actual number of tokens (trg_len) to get the total Loss.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # 4. Calculate Perplexity for the entire text
    # Perplexity = exp(total Loss / total tokens)
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    
    print("Perplexity calculation completed.")
    return ppl.item()

