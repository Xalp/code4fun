import json

def load_responses(filename):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            item = json.loads(line)
            data[item['doc_id']] = item
    return data

def save_divergent_examples():
    temp0 = load_responses("gsm8k_temp0.jsonl")
    temp1 = load_responses("gsm8k_temp1.jsonl")

    print(f"Loaded {len(temp0)} responses for Temp0 and {len(temp1)} responses for Temp1.")
    
    wins = []
    
    for doc_id, t1_item in temp1.items():
        if doc_id not in temp0:
            continue
            
        t0_item = temp0[doc_id]
        
        # Check conditions: temp1 Correct (1), temp0 Wrong (0)
        t1_score = t1_item.get('math_verify', 0)
        t0_score = t0_item.get('math_verify', 0)

        print(f"Doc {doc_id}: Temp1 score {t1_score}, Temp0 score {t0_score}")
        
        if t1_score == 1 and t0_score == 0:
            entry = {
                "doc_id": doc_id,
                "problem": t1_item['doc']['problem'],
                "solution": t1_item['doc']['solution'],
                "target": t1_item['target'],
                "temp1_ans": t1_item.get('filtered_resps', [""])[0],
                "temp0_ans": t0_item.get('filtered_resps', [""])[0],
                "temp1_full_resp": t1_item.get('resps', [[""]])[0][0],
                "temp0_full_resp": t0_item.get('resps', [[""]])[0][0]
            }
            wins.append(entry)
            
    print(f"Found {len(wins)} divergent examples (Temp1 Win / Temp0 Loss).")
    
    with open("ablation_win.jsonl", 'w') as f:
        for win in wins:
            f.write(json.dumps(win) + '\n')
    
    print("Saved to ablation_win.jsonl")

if __name__ == "__main__":
    save_divergent_examples()
