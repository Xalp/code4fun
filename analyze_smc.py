import json
import glob
import numpy as np
import os

def analyze_all_files(root_dir="."):
    search_pattern = os.path.join(root_dir, "**", "smc_trace*.jsonl")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(files)} trace files in {root_dir}.")
    
    global_overtake_count = 0
    global_total_blocks = 0
    
    for filename in files:
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
            
            # For Block 0, Start is implicitly uniform (all equal). No overtake possible by definition 
            # (or all considered dominant). We skip Block 0 for "Overtake" stats usually, or count 0.
            # Let's iterate from Block 0 to N.
            # Block 0: Start Probs = Uniform. Max = Uniform. All are dominant. 
            #   End Probs = traced. Winner w. Was w dominant at start? Yes. So no overtake.
            
            for i in range(len(sample)):
                block_data = sample[i]
                final_log_probs = np.array(block_data['log_probs'])
                
                # Derive Start Probs
                if i == 0:
                    # Block 0: Assume uniform start (all 0 log prob relative to each other)
                    start_log_probs = np.zeros_like(final_log_probs)
                else:
                    # Start of Block i comes from End of Block i-1 Resampled
                    prev_block = sample[i-1]
                    prev_final_probs = np.array(prev_block['log_probs'])
                    selected_indices = prev_block.get('selected_indices')
                    
                    if selected_indices:
                        # Map parents to children
                        start_log_probs = prev_final_probs[selected_indices]
                    else:
                        start_log_probs = prev_final_probs
                
                # Define Dominance
                # Using a small epsilon for float comparison safety, though they usually come from same values
                max_start = np.max(start_log_probs)
                is_dominant_start = start_log_probs >= (max_start - 1e-6)
                
                max_end = np.max(final_log_probs)
                is_dominant_end = final_log_probs >= (max_end - 1e-6)
                
                # Check for Overtake
                # An overtake happens if there exists a particle k such that:
                # k is dominant at END, BUT k was NOT dominant at START.
                
                # Get indices of winners at end
                winners = np.where(is_dominant_end)[0]
                
                overtake_happened = False
                for w in winners:
                    if not is_dominant_start[w]:
                        overtake_happened = True
                        break
                
                if overtake_happened:
                    global_overtake_count += 1
                
                global_total_blocks += 1
            
    print("="*40)
    print("GLOBAL AGGREGATED RESULTS (PER BLOCK)")
    print(f"Total files processed: {len(files)}")
    print(f"Total blocks analyzed: {global_total_blocks}")
    if global_total_blocks > 0:
        pct = (global_overtake_count / global_total_blocks) * 100
        print(f"Overtake Events: {global_overtake_count} ({pct:.2f}%)")
        print("Note: An overtake is defined as a block where the generation winner started as a non-dominant particle.")
    else:
        print("No blocks found.")
    print("="*40)

if __name__ == "__main__":
    analyze_all_files()
