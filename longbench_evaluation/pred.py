#!/usr/bin/env python3
"""
LongBench v2 Evaluation Script with Hugging Face Transformers
Modified from the official LongBench repository to use HuggingFace instead of vLLM
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import time

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
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        do_sample=False,  # temperature, top_p 제거
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

def predict_samples(model, tokenizer, config, data, max_length, max_gen, cot, no_context, rag_k):
    """Prediction function following original structure"""
    results = []
    
    for sample in tqdm(data, desc="Processing"):
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
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {sample['_id']}: {str(e)}")
            # Add error result
            result = {
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
            results.append(result)
            continue
    
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
    
    predictions = predict_samples(
        model=model,
        tokenizer=tokenizer, 
        config=config,
        data=dataset,
        max_length=max_length,
        max_gen=args.max_gen,
        cot=args.cot,
        no_context=args.no_context,
        rag_k=args.rag
    )
    
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    
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