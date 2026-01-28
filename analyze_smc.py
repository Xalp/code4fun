import json
import glob
import numpy as np

def analyze_file(filename):
    print(f"Analyzing {filename}...")
    with open(filename, 'r') as f:
        lines = [json.loads(line) for line in f]
    
    # Organize by sample (not explicitly tracked in trace, but PID separates traces if 1 sample/run, 
    # but accelerate might run multiple samples in one PID seq).
    # Since num_block resets, we can detect new samples.
    
    samples = []
    current_sample = []
    
    for line in lines:
        if line['num_block'] == 0 and current_sample:
            samples.append(current_sample)
            current_sample = []
        current_sample.append(line)
    if current_sample:
        samples.append(current_sample)
        
    print(f"Found {len(samples)} generation traces.")
    
    overtake_count = 0
    total_resamples = 0
    
    # Stats
    dominant_lost_count = 0 # Initially dominant (rank 0) lost leadership or died
    rank_changes = 0
    
    for sample in samples:
        # analyzing one generation flow
        initial_log_probs = sample[0]['log_probs']
        initial_dominant_idx = np.argmax(initial_log_probs)
        initial_dominant_id = sample[0]['particle_ids'][initial_dominant_idx]
        
        final_log_probs = sample[-1]['log_probs']
        final_dominant_idx = np.argmax(final_log_probs)
        final_dominant_id = sample[-1]['particle_ids'][final_dominant_idx]
        
        # Check if the final winner is the same as the initial winner
        if initial_dominant_id != final_dominant_id:
            overtake_count += 1
            
        # Check step-by-step resampling
        for i in range(len(sample)-1):
            step_data = sample[i]
            if step_data['resampled']:
                total_resamples += 1
                
                # Check if the dominant particle (highest weight/prob) was preserved
                # Weights determine resampling. Dominant usually means highest weight.
                weights = step_data.get('log_weights') 
                # Note: log_weights might be normalized or raw. generate_smc uses log_w.
                # Assuming highest log_weight is "dominant" candidate for resampling.
                
                # Check lineage
                # If the max-prob particle at step i is NOT the ancestor of the max-prob particle at step i+1
                pass

    print(f"Total samples: {len(samples)}")
    print(f"Number of times the final winner was NOT the initially dominant particle: {overtake_count} ({overtake_count/len(samples)*100:.2f}%)")

if __name__ == "__main__":
    files = glob.glob("./smc_trace_pid*.jsonl")
    for f in files:
        analyze_file(f)
