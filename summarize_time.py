import glob
import re
import os

def parse_log(filepath):
    # Regex to find time
    # matches: "Total time taken: 123.45 seconds" (LLaDA)
    # matches: "Time taken: 123.45 seconds" (Dream)
    time_taken = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if "Time taken:" in line:
                try:
                    # Extract float
                    val = float(re.search(r"taken: ([\d\.]+) seconds", line).group(1))
                    time_taken.append(val)
                except:
                    pass
            elif "Total time taken:" in line:
                try:
                    val = float(re.search(r"taken: ([\d\.]+)", line).group(1))
                    time_taken.append(val)
                except:
                    pass
                    
    if not time_taken:
        return 0.0
    
    # In distributed setting, multiple ranks print time. 
    # The actual end-to-end latency is roughly the max of them (wait for all).
    return max(time_taken)

def summarize():
    logs = glob.glob("timing_logs/*.log")
    results = {}
    
    for log in logs:
        model = os.path.basename(log).replace(".log", "")
        duration = parse_log(log)
        results[model] = duration
        
    print(f"{'Experiment':<20} | {'Wall Time (s)':<15} | {'GPU Steps (est)':<15}")
    print("-" * 55)
    
    # Sort for consistent output
    for key in sorted(results.keys()):
        val = results[key]
        print(f"{key:<20} | {val:.2f}            | {val*8:.2f} (if summing 8 GPUs)")

if __name__ == "__main__":
    summarize()
