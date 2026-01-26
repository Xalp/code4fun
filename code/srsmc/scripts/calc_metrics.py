import argparse
import json
import os
import glob
import re
from typing import List, Dict

def read_jsonl(path: str):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode line in {path}")
    return data

def extract_code(text: str) -> str:
    """Extract python code block from text (similar to LLaDA utils)"""
    if "```python" in text:
        text = text.split("```python")[1]
        if "```" in text:
            text = text.split("```")[0]
    elif "```" in text:
        text = text.split("```")[1] 
        if "```" in text:
            text = text.split("```")[0]
    return text.strip()

def pass_at_1(predictions, references):
    """Simple pass@1 estimator. 
    Note: Real execution requires the 'evaluate' library and a sandbox. 
    This is a PLACEHOLDER for offline metric calculation if we don't want to run dangerous code here.
    However, the user asked for offline eval. 
    For Math/GSM8K we can check string match.
    For Code, we typically need to run it.
    
    If the saved output contains 'OOM', it counts as fail.
    """
    correct = 0
    total = len(predictions)
    
    # Ideally we use lm_eval's metrics, but they are coupled with the library.
    # For now, we will print the count of non-OOM answers.
    # To truly run code eval offline, we need to import the task specific logic.
    pass

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from saved generation files.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory containing rank_*.jsonl files")
    parser.add_argument("--task", type=str, required=True, choices=['mbpp', 'humaneval', 'gsm8k', 'math500'], help="Task name")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.save_dir, "rank_*.jsonl"))
    if not files:
        print(f"No rank_*.jsonl files found in {args.save_dir}")
        return

    print(f"Found {len(files)} files: {files}")
    
    all_results = []
    for f in files:
        data = read_jsonl(f)
        all_results.extend(data)
    
    print(f"Total samples loaded: {len(all_results)}")
    
    # Filter out OOMs
    valid_results = [r for r in all_results if r != "OOM" and r != "# OOM: skipped"]
    oom_count = len(all_results) - len(valid_results)
    
    print(f"OOM Failures: {oom_count}")
    print(f"Valid Outputs: {len(valid_results)}")

    if args.task in ['gsm8k', 'math500']:
        print("Metric calculation for Math requires ground truth which is usually inside the prompt or dataset.")
        print("Since saved files only contain answers (usually), we might need the full request data.")
        print("NOTE: The current 'save_dir' logic in eval_llada.py only saves the *generated answer string*.")
        print("To verify correctness offline, we would need the associated target/gold answer.")
        print("Recommendation: Use this script to check completion status. To get metrics, resumption of the main eval script is best as it compares against the loaded dataset.")
    
    elif args.task in ['mbpp', 'humaneval']:
        print("For code tasks, you typically need to execute the code.")
        print("This script currently confirms that you have generated outputs.")
        print("Pass@1 calculation requires execution against test cases.")
        
        # We can try to use lm_eval to evaluate these offline if we construct a dummy model
        # But for now, we report the counts.
        
    print("="*30)
    print("Generation Status Check Complete")
    print(f"You have {len(valid_results)} usable generations.")
    if oom_count > 0:
        print(f"WARNING: {oom_count} samples failed due to OOM.")

if __name__ == "__main__":
    main()
