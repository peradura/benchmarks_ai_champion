#!/usr/bin/env python3
"""
Result calculation script for LongBench v2
Compatible with the official evaluation format
"""

import json
import os
import argparse
from collections import defaultdict
from datasets import load_dataset

def load_predictions(pred_file):
    """Load predictions from jsonl file"""
    predictions = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pred = json.loads(line)
                predictions[pred['_id']] = pred['pred']
    return predictions

def calculate_accuracy(predictions, ground_truth):
    """Calculate accuracy"""
    correct = 0
    total = 0
    
    for _id, gt_answer in ground_truth.items():
        if _id in predictions:
            pred_answer = predictions[_id]
            if pred_answer == gt_answer:
                correct += 1
            total += 1
        else:
            print(f"Warning: Missing prediction for {_id}")
    
    if total == 0:
        return 0.0
    
    return correct / total * 100

def calculate_detailed_results(predictions, dataset):
    """Calculate detailed results by category"""
    results = defaultdict(lambda: {"correct": 0, "total": 0})
    ground_truth = {}
    
    # Build ground truth and categorize
    for sample in dataset:
        _id = sample["_id"]
        ground_truth[_id] = sample["answer"]
        
        # Categories for analysis
        categories = [
            ("domain", sample["domain"]),
            ("sub_domain", sample["sub_domain"]), 
            ("difficulty", sample["difficulty"]),
            ("length", sample["length"])
        ]
        
        for cat_type, cat_value in categories:
            key = f"{cat_type}_{cat_value}"
            results[key]["total"] += 1
            
            if _id in predictions and predictions[_id] == sample["answer"]:
                results[key]["correct"] += 1
    
    # Calculate accuracies
    accuracy_results = {}
    for key, stats in results.items():
        if stats["total"] > 0:
            accuracy_results[key] = {
                "accuracy": stats["correct"] / stats["total"] * 100,
                "correct": stats["correct"],
                "total": stats["total"]
            }
    
    # Overall accuracy
    overall_correct = sum(stats["correct"] for stats in results.values() 
                         if stats["total"] > 0)
    overall_total = len(ground_truth)
    overall_accuracy = overall_correct / overall_total * 100 if overall_total > 0 else 0
    
    return {
        "overall": {
            "accuracy": overall_accuracy,
            "correct": overall_correct, 
            "total": overall_total
        },
        "by_category": accuracy_results,
        "ground_truth": ground_truth
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default="pred", 
                       help="Directory containing prediction files")
    parser.add_argument("--model", type=str, help="Specific model to evaluate")
    parser.add_argument("--output", type=str, default="result.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Load dataset for ground truth
    print("Loading LongBench v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    
    if args.model:
        # Evaluate specific model
        model_dir = os.path.join(args.pred_dir, args.model.replace('/', '_'))
        if not os.path.exists(model_dir):
            print(f"Error: Directory {model_dir} not found")
            return
        
        pred_files = [f for f in os.listdir(model_dir) if f.endswith('.jsonl')]
        if not pred_files:
            print(f"Error: No .jsonl files found in {model_dir}")
            return
            
        results = {}
        for pred_file in pred_files:
            pred_path = os.path.join(model_dir, pred_file)
            setting_name = pred_file[:-6]  # Remove .jsonl
            
            print(f"Evaluating {setting_name}...")
            predictions = load_predictions(pred_path)
            detailed_results = calculate_detailed_results(predictions, dataset)
            results[setting_name] = detailed_results
            
            print(f"Overall accuracy: {detailed_results['overall']['accuracy']:.2f}%")
            
        # Save results
        output_path = os.path.join(model_dir, args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"Results saved to: {output_path}")
        
    else:
        # Evaluate all models
        if not os.path.exists(args.pred_dir):
            print(f"Error: Directory {args.pred_dir} not found")
            return
            
        all_results = {}
        
        for model_name in os.listdir(args.pred_dir):
            model_dir = os.path.join(args.pred_dir, model_name)
            if not os.path.isdir(model_dir):
                continue
                
            pred_files = [f for f in os.listdir(model_dir) if f.endswith('.jsonl')]
            if not pred_files:
                continue
                
            print(f"\nEvaluating model: {model_name}")
            model_results = {}
            
            for pred_file in pred_files:
                pred_path = os.path.join(model_dir, pred_file)
                setting_name = pred_file[:-6]  # Remove .jsonl
                
                print(f"  Setting: {setting_name}")
                predictions = load_predictions(pred_path)
                detailed_results = calculate_detailed_results(predictions, dataset)
                model_results[setting_name] = detailed_results
                
                print(f"    Accuracy: {detailed_results['overall']['accuracy']:.2f}%")
            
            all_results[model_name] = model_results
        
        # Save consolidated results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
            
        print(f"\nConsolidated results saved to: {args.output}")

if __name__ == "__main__":
    main()