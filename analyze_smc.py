import json
import glob
import numpy as np
import os

def analyze_all_files(root_dir="."):
    search_pattern = os.path.join(root_dir, "**", "smc_trace*.jsonl")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} trace files in {root_dir}.")
    
    global_overtake_count = 0
    global_total_samples = 0
    
    for filename in files:
        # print(f"Analyzing {filename}...")
        try:
            with open(filename, 'r') as f:
                lines = [json.loads(line) for line in f]
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        if not lines:
            continue
            
        samples = []
        current_sample = []
        
        for line in lines:
            if line['num_block'] == 0 and current_sample:
                samples.append(current_sample)
                current_sample = []
            current_sample.append(line)
        if current_sample:
            samples.append(current_sample)
            
        # Analyze current file samples
        for sample in samples:
            if not sample: continue
            
            initial_log_probs = sample[0]['log_probs']
            initial_dominant_idx = np.argmax(initial_log_probs)
            initial_dominant_id = sample[0]['particle_ids'][initial_dominant_idx]
            
            final_log_probs = sample[-1]['log_probs']
            final_dominant_idx = np.argmax(final_log_probs)
            final_dominant_id = sample[-1]['particle_ids'][final_dominant_idx]
            
            # Check if the final winner is the same as the initial winner
            if initial_dominant_id != final_dominant_id:
                global_overtake_count += 1
                
            global_total_samples += 1
            
    print("="*40)
    print("GLOBAL AGGREGATED RESULTS")
    print(f"Total files processed: {len(files)}")
    print(f"Total samples collected: {global_total_samples}")
    if global_total_samples > 0:
        pct = (global_overtake_count / global_total_samples) * 100
        print(f"Number of times the final winner was NOT the initially dominant particle: {global_overtake_count} ({pct:.2f}%)")
    else:
        print("No samples found.")
    print("="*40)

if __name__ == "__main__":
    analyze_all_files()
