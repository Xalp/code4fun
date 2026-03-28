"""
Aggregate N-seed runs to compute Majority Voting and Pass@N accuracy.
Reads lm-eval output files from results_rebuttal/{model}/{task}/seed{1..N}/

Usage:
    python aggregate_seeds.py --results_dir ./results_rebuttal --num_seeds 4
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


def extract_gsm8k_answer(text):
    """Extract numeric answer from GSM8K-style output (after ####)."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    # Fallback: last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else ""


def extract_math_answer(text):
    """Extract answer from MATH-style output (\\boxed{...})."""
    match = re.search(r'\\boxed\{([^}]*)\}', text)
    if match:
        return match.group(1).strip()
    return text.strip().split('\n')[-1].strip()


def extract_code_answer(text):
    """For code tasks, use the full text as the 'answer'."""
    return text.strip()


EXTRACTORS = {
    'gsm8k': extract_gsm8k_answer,
    'math500': extract_math_answer,
    'mbpp': extract_code_answer,
    'humaneval': extract_code_answer,
}


def load_seed_results(results_dir, model, task, num_seeds):
    """Load per-sample results from each seed run."""
    all_seeds = []
    for seed in range(1, num_seeds + 1):
        seed_dir = Path(results_dir) / model / task / f"seed{seed}"
        # Find the results json from lm-eval
        result_files = list(seed_dir.rglob("results*.json"))
        sample_files = list(seed_dir.rglob("samples_*.jsonl"))

        if not sample_files:
            print(f"  WARNING: No sample files in {seed_dir}, skipping seed {seed}")
            continue

        samples = []
        for sf in sorted(sample_files):
            with open(sf) as f:
                for line in f:
                    samples.append(json.loads(line))
        all_seeds.append(samples)

    return all_seeds


def compute_metrics(all_seeds, task):
    """Compute Pass@N and Majority Voting accuracy."""
    if not all_seeds:
        return {}

    num_seeds = len(all_seeds)
    num_samples = len(all_seeds[0])

    pass_at_n_correct = 0
    maj_vote_correct = 0
    total = 0

    for i in range(num_samples):
        # Check if each seed has this sample
        seed_results = []
        for s in range(num_seeds):
            if i < len(all_seeds[s]):
                seed_results.append(all_seeds[s][i])

        if not seed_results:
            continue

        total += 1

        # Get correctness of each seed (lm-eval stores this)
        correctness = []
        answers = []
        for sr in seed_results:
            # lm-eval stores filtered results
            if 'filtered_resps' in sr:
                resp = sr['filtered_resps'][0] if sr['filtered_resps'] else ""
            elif 'resps' in sr:
                resp = sr['resps'][0][0] if sr['resps'] else ""
            else:
                resp = sr.get('model_output', sr.get('response', ''))

            # Check if marked correct by lm-eval
            is_correct = False
            # Try all known correctness keys
            for key in ['exact_match', 'acc', 'pass@1', 'pass_at_1',
                        'exact_match,strict-match', 'exact_match,flexible-extract',
                        'exact_match,none']:
                if key in sr:
                    val = sr[key]
                    if isinstance(val, (int, float)) and val > 0:
                        is_correct = True
                        break

            correctness.append(is_correct)
            answers.append(str(resp).strip())

        # Pass@N: any seed correct
        if any(correctness):
            pass_at_n_correct += 1

        # Majority Voting: vote on most common answer
        if answers:
            vote = Counter(answers).most_common(1)[0][0]
            # Check if the voted answer matches any correct answer
            for j, ans in enumerate(answers):
                if ans == vote and correctness[j]:
                    maj_vote_correct += 1
                    break

    return {
        'total': total,
        'num_seeds': num_seeds,
        f'pass@{num_seeds}': pass_at_n_correct / total * 100 if total > 0 else 0,
        'majority_vote': maj_vote_correct / total * 100 if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='./results_rebuttal')
    parser.add_argument('--num_seeds', type=int, default=4)
    args = parser.parse_args()

    models = ['llada1.5', 'dream']
    tasks = ['gsm8k', 'math500', 'mbpp', 'humaneval']

    print(f"{'Model':<12} {'Task':<12} {'Pass@{}'.format(args.num_seeds):<12} {'MajVote':<12} {'N_samples':<10}")
    print("-" * 58)

    for model in models:
        for task in tasks:
            all_seeds = load_seed_results(args.results_dir, model, task, args.num_seeds)
            if not all_seeds:
                print(f"{model:<12} {task:<12} {'N/A':<12} {'N/A':<12}")
                continue
            metrics = compute_metrics(all_seeds, task)
            print(f"{model:<12} {task:<12} {metrics.get(f'pass@{args.num_seeds}', 0):<12.1f} {metrics.get('majority_vote', 0):<12.1f} {metrics.get('total', 0):<10}")


if __name__ == '__main__':
    main()
