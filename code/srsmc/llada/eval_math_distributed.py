"""
Distributed SMC evaluation on MATH (minerva_math500).
Launch: torchrun --nproc_per_node=8 eval_math_distributed.py [--mode smc|bon] [--local_particles 8]

Produces: Pass@N, BoN@N, SR-SMC@N results in one run.
"""
import argparse
import json
import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from model.modeling_llada import LLaDAModelLM
from generate_smc_distributed import generate_distributed_smc

# Math answer extraction (from lm-eval)
def extract_answer(text):
    """Extract answer from \\boxed{} or last number."""
    # Try boxed
    match = re.findall(r'\\boxed\{([^}]*)\}', text)
    if match:
        return match[-1].strip()
    # Try #### pattern
    match = re.search(r'####\s*(.+)', text)
    if match:
        return match.group(1).strip()
    # Last number
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()


def check_math_answer(pred, target):
    """Simple math answer checking."""
    pred = pred.strip().rstrip('.')
    target = target.strip().rstrip('.')
    if pred == target:
        return True
    try:
        return abs(float(pred) - float(target)) < 1e-6
    except:
        return False


def build_few_shot_prompt(tokenizer, question, examples, n_shot=4):
    """Build n-shot prompt for MATH."""
    prompt_parts = []
    for ex in examples[:n_shot]:
        prompt_parts.append(f"Problem: {ex['problem']}\nSolution: {ex['solution']}")
    prompt_parts.append(f"Problem: {question}\nSolution:")
    prompt_text = "\n\n".join(prompt_parts)

    messages = [{"role": "user", "content": prompt_text}]
    chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return chat_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smc', 'bon'], default='smc',
                        help='smc = distributed SMC with resampling; bon = no resampling (independent samples)')
    parser.add_argument('--local_particles', type=int, default=8)
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--n_shot', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./results_distributed')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    # Init distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    total_particles = args.local_particles * world_size
    resample_strategy = "adaptive" if args.mode == "smc" else "never"

    if rank == 0:
        print(f"=== Distributed SMC Eval ===")
        print(f"Mode: {args.mode}, {world_size} GPUs × {args.local_particles} particles = {total_particles} total")
        print(f"Gen: length={args.gen_length}, block={args.block_length}, temp={args.temperature}")

    # Load model
    model_path = "GSAI-ML/LLaDA-1.5"
    mask_id = 126336
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config,
                                          device_map={'': device}).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load MATH dataset
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_mathematics", split="test", trust_remote_code=True)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Few-shot examples from train split
    train_ds = load_dataset("hendrycks/competition_mathematics", split="train", trust_remote_code=True)
    few_shot_examples = [train_ds[i] for i in range(args.n_shot)]

    steps = args.gen_length // args.block_length

    results = []
    correct_smc = 0
    correct_bon = 0
    correct_any = 0
    total = 0

    for idx in tqdm(range(len(ds)), disable=(rank != 0)):
        item = ds[idx]
        question = item['problem']
        target = item['solution']
        # Extract target answer
        target_answer = extract_answer(target)

        # Build prompt
        prompt_text = build_few_shot_prompt(tokenizer, question, few_shot_examples, args.n_shot)
        input_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'].to(device)

        # Run distributed SMC
        all_x, all_logp, nfe = generate_distributed_smc(
            model, input_ids, steps=steps, gen_length=args.gen_length,
            block_length=args.block_length, temperature=args.temperature,
            remasking='low_confidence', mask_id=mask_id, threshold=0.9,
            local_particles=args.local_particles, resample_strategy=resample_strategy)

        # Only rank 0 evaluates
        if rank == 0:
            total += 1
            particle_results = []
            for p in range(all_x.shape[0]):
                gen_text = tokenizer.decode(all_x[p, input_ids.shape[1]:], skip_special_tokens=True)
                pred_answer = extract_answer(gen_text)
                is_correct = check_math_answer(pred_answer, target_answer)
                p_logp = all_logp[p].sum().item()
                particle_results.append({
                    'particle': p,
                    'logp': p_logp,
                    'answer': pred_answer,
                    'correct': is_correct,
                    'text': gen_text[:200],
                })

            # SR-SMC: best by logp (this is what SMC returns)
            best_idx = max(range(len(particle_results)), key=lambda i: particle_results[i]['logp'])
            smc_correct = particle_results[best_idx]['correct']

            # BoN: same as SMC selection (best logp)
            bon_correct = smc_correct

            # Pass@N: any correct
            any_correct = any(pr['correct'] for pr in particle_results)

            correct_smc += smc_correct
            correct_bon += bon_correct
            correct_any += any_correct

            results.append({
                'idx': idx,
                'question': question[:100],
                'target': target_answer,
                'smc_correct': smc_correct,
                'bon_correct': bon_correct,
                'any_correct': any_correct,
                'particles': particle_results,
            })

            if total % 50 == 0:
                print(f"[{total}/{len(ds)}] SMC={correct_smc/total*100:.1f}% "
                      f"BoN={correct_bon/total*100:.1f}% Pass@{total_particles}={correct_any/total*100:.1f}%")

        # Sync all ranks before next question
        dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Final Results (N={total_particles}, {total} questions)")
        print(f"{'='*60}")
        print(f"SR-SMC@{total_particles}:  {correct_smc/total*100:.2f}%")
        print(f"BoN@{total_particles}:     {correct_bon/total*100:.2f}%")
        print(f"Pass@{total_particles}:    {correct_any/total*100:.2f}%")

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'{args.mode}_{total_particles}particles.json'), 'w') as f:
            json.dump({
                'mode': args.mode,
                'total_particles': total_particles,
                'local_particles': args.local_particles,
                'world_size': world_size,
                'total_questions': total,
                'smc_accuracy': correct_smc / total * 100,
                'bon_accuracy': correct_bon / total * 100,
                'pass_at_n': correct_any / total * 100,
                'results': results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output_dir}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
