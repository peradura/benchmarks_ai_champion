#!/usr/bin/env python3
import os
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp

# Import config module (create this to match the original structure)
try:
    from config import MODEL_CONFIG
except ImportError:
    MODEL_CONFIG = {}

def get_model_config(model_name):
    """Get model configuration from config file"""
    return MODEL_CONFIG.get(model_name, {
        "max_model_len": 32768,
        "model_path": model_name
    })

def load_model_and_tokenizer(model_name, device_map="auto", torch_dtype=torch.bfloat16):
    """Load model and tokenizer"""
    config = get_model_config(model_name)
    model_path = config.get("model_path", model_name)
    
    print(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    model.eval()
    return model, tokenizer, config

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """Generate response from model with thread safety"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Use torch.cuda.stream for better GPU utilization if available
        if torch.cuda.is_available():
            with torch.cuda.device(model.device):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    # Extract generated part only
    generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()

def build_chat(context, question, choices, cot=False):
    """Build chat prompt following the official format"""
    if cot:
        instruction = "Please read the following context and answer the question step by step. Think through your reasoning before providing the final answer."
    else:
        instruction = "Please read the following context and answer the question."
    
    prompt = f"""{instruction}

Context: {context}

Question: {question}

{choices}

Please select the correct answer (A, B, C, or D):"""
    
    return prompt

def truncate_input(tokenizer, context, question, choices, max_length):
    """Truncate input following the original method"""
    # Build full prompt to check length
    full_prompt = build_chat(context, question, choices)
    tokens = tokenizer.encode(full_prompt)
    
    if len(tokens) <= max_length:
        return context
    
    # Calculate available length for context
    question_choices = f"Question: {question}\n\n{choices}\n\nPlease select the correct answer (A, B, C, or D):"
    overhead_tokens = len(tokenizer.encode(question_choices)) + 100  # Some buffer
    available_length = max_length - overhead_tokens
    
    if available_length <= 0:
        return ""
    
    # Truncate from middle (keep beginning and end)
    context_tokens = tokenizer.encode(context)
    if len(context_tokens) <= available_length:
        return context
    
    # Keep first and last parts
    keep_length = available_length // 2
    truncated_tokens = context_tokens[:keep_length] + context_tokens[-keep_length:]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)

def process_single_sample(args_tuple):
    """Process a single sample - for parallel processing"""
    sample, model, tokenizer, max_length, max_gen, cot, no_context, rag_k = args_tuple
    
    context = sample["context"] if not no_context else ""
    question = sample["question"]
    
    # Build choices
    choices = f"A. {sample['choice_A']}\nB. {sample['choice_B']}\nC. {sample['choice_C']}\nD. {sample['choice_D']}"
    
    # Handle RAG if needed
    if rag_k > 0:
        # RAG implementation would go here
        # For now, use full context
        pass
    
    # Truncate if necessary
    if context:
        context = truncate_input(tokenizer, context, question, choices, max_length)
    
    # Build prompt
    prompt = build_chat(context, question, choices, cot)
    
    try:
        # Generate response
        response = generate_response(model, tokenizer, prompt, max_gen)
        
        # Extract answer
        pred_answer = None
        for choice in ['A', 'B', 'C', 'D']:
            if choice in response.upper():
                pred_answer = choice
                break
        
        if pred_answer is None:
            pred_answer = "A"  # Default fallback
        
        result = {
            "_id": sample["_id"],
            "domain": sample["domain"],
            "sub_domain": sample["sub_domain"],  
            "difficulty": sample["difficulty"],
            "length": sample["length"],
            "context": context,
            "question": question,
            "choices": choices,
            "pred": pred_answer,
            "response": response
        }
        return result, None
        
    except Exception as e:
        error_result = {
            "_id": sample["_id"],
            "domain": sample["domain"],
            "sub_domain": sample["sub_domain"],
            "difficulty": sample["difficulty"], 
            "length": sample["length"],
            "context": context,
            "question": question,
            "choices": choices,
            "pred": "A",  # Default
            "response": "ERROR"
        }
        return error_result, str(e)
def predict_samples(model, tokenizer, config, data, max_length, max_gen, cot, no_context, rag_k, num_workers=4):
    """Prediction function with parallel processing"""
    results = []
    lock = Lock()
    
    # Determine optimal number of workers
    if num_workers == -1:
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print(f"Using {num_workers} workers for parallel processing")
    
    # Prepare arguments for parallel processing
    args_list = [
        (sample, model, tokenizer, max_length, max_gen, cot, no_context, rag_k)
        for sample in data
    ]
    
    # Use ThreadPoolExecutor for I/O bound tasks (GPU inference)
    # For CPU-heavy tasks, you might want to use ProcessPoolExecutor instead
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Configure tqdm to stay on one line
        pbar = tqdm(total=len(data), desc="Processing", ncols=100, file=sys.stdout, leave=True)
        
        # Submit all tasks
        future_to_sample = {
            executor.submit(process_single_sample, args): args[0]
            for args in args_list
        }
        
        error_count = 0
        
        # Process completed tasks
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            
            try:
                result, error = future.result()
                
                with lock:
                    results.append(result)
                    if error:
                        error_count += 1
                        print(f"\nError processing sample {sample['_id']}: {error}")
                    
                    # Update progress bar description with current progress
                    pbar.set_description(f"Processing [{len(results)}/{len(data)}] (Errors: {error_count})")
                    pbar.update(1)
                    
            except Exception as e:
                with lock:
                    error_count += 1
                    print(f"\nUnexpected error with sample {sample['_id']}: {str(e)}")
                    # Add default error result
                    error_result = {
                        "_id": sample["_id"],
                        "domain": sample.get("domain", "unknown"),
                        "sub_domain": sample.get("sub_domain", "unknown"),
                        "difficulty": sample.get("difficulty", "unknown"), 
                        "length": sample.get("length", "unknown"),
                        "context": "",
                        "question": sample.get("question", ""),
                        "choices": "",
                        "pred": "A",
                        "response": "CRITICAL_ERROR"
                    }
                    results.append(error_result)
                    pbar.set_description(f"Processing [{len(results)}/{len(data)}] (Errors: {error_count})")
                    pbar.update(1)
        
        pbar.close()
    
    # Sort results by original order (by _id) to maintain consistency
    results.sort(key=lambda x: x["_id"])
    
    if error_count > 0:
        print(f"\nTotal errors encountered: {error_count}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--cot", action="store_true", help="Enable Chain-of-Thought")
    parser.add_argument("--no_context", action="store_true", help="Test without context")
    parser.add_argument("--rag", type=int, default=0, help="RAG top-k (0 to disable)")
    parser.add_argument("--max_gen", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--device", type=str, default="auto", help="Device mapping")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", 
                       choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of parallel workers (-1 for auto)")
    parser.add_argument("--no_parallel", action="store_true", 
                       help="Disable parallel processing")
    
    args = parser.parse_args()
    
    # Convert torch_dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, config = load_model_and_tokenizer(
        args.model, 
        device_map=args.device,
        torch_dtype=torch_dtype
    )
    
    max_length = config.get("max_model_len", 32768)
    print(f"Model max length: {max_length}")
    
    # Load data
    print("Loading LongBench v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    
    # Run prediction
    print("Starting evaluation...")
    start_time = time.time()
    
    # Determine number of workers
    num_workers = 1 if args.no_parallel else args.num_workers
    
    predictions = predict_samples(
        model=model,
        tokenizer=tokenizer, 
        config=config,
        data=dataset,
        max_length=max_length,
        max_gen=args.max_gen,
        cot=args.cot,
        no_context=args.no_context,
        rag_k=args.rag,
        num_workers=num_workers
    )
    
    end_time = time.time()
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    output_dir = f"pred/{args.model.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build filename
    filename_parts = [args.model.replace('/', '_')]
    if args.cot:
        filename_parts.append("cot")
    if args.no_context:
        filename_parts.append("no_context")
    if args.rag > 0:
        filename_parts.append(f"rag{args.rag}")
    
    filename = "_".join(filename_parts) + ".jsonl"
    output_path = os.path.join(output_dir, filename)
    
    # Save predictions
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    print(f"Results saved to: {output_path}")
    
    # Basic stats
    total = len(predictions)
    print(f"Total samples processed: {total}")
    
    # Clean up
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()