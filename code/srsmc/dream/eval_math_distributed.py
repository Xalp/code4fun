"""
Distributed SMC evaluation on MATH for Dream-7B.
Launch: torchrun --nproc_per_node=8 eval_math_distributed.py [--mode smc|bon] [--local_particles 8]
"""
import argparse
import json
import os
import re
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer
from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from generate_smc_distributed import generate_distributed_smc


def extract_answer(text):
    match = re.findall(r'\\boxed\{([^}]*)\}', text)
    if match:
        return match[-1].strip()
    match = re.search(r'####\s*(.+)', text)
    if match:
        return match.group(1).strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else text.strip()


def check_math_answer(pred, target):
    pred = pred.strip().rstrip('.')
    target = target.strip().rstrip('.')
    if pred == target:
        return True
    try:
        return abs(float(pred) - float(target)) < 1e-6
    except:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smc', 'bon'], default='smc')
    parser.add_argument('--local_particles', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--n_shot', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./results_distributed')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    total_particles = args.local_particles * world_size
    resample_strategy = "adaptive" if args.mode == "smc" else "never"
    steps = args.max_new_tokens // args.block_length

    if rank == 0:
        print(f"=== Dream-7B Distributed SMC ===")
        print(f"Mode: {args.mode}, {world_size} GPUs × {args.local_particles} = {total_particles} particles")

    # Load model
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                        trust_remote_code=True, device_map={'': device}).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load MATH
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_mathematics", split="test", trust_remote_code=True)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    train_ds = load_dataset("hendrycks/competition_mathematics", split="train", trust_remote_code=True)
    few_shot_examples = [train_ds[i] for i in range(args.n_shot)]

    results = []
    correct_smc = 0
    correct_any = 0
    total = 0

    for idx in tqdm(range(len(ds)), disable=(rank != 0)):
        item = ds[idx]
        question = item['problem']
        target = item['solution']
        target_answer = extract_answer(target)

        # Build prompt
        prompt_parts = []
        for ex in few_shot_examples[:args.n_shot]:
            prompt_parts.append(f"Problem: {ex['problem']}\nSolution: {ex['solution']}")
        prompt_parts.append(f"Problem: {question}\nSolution:")
        prompt_text = "\n\n".join(prompt_parts)
        messages = [{"role": "user", "content": prompt_text}]
        chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(chat_text, return_tensors='pt').input_ids.to(device)
        attn_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

        all_x, all_logp = generate_distributed_smc(
            model, input_ids, attention_mask=attn_mask,
            steps=steps, max_new_tokens=args.max_new_tokens,
            block_length=args.block_length, temperature=args.temperature,
            threshold=0.9, local_particles=args.local_particles,
            resample_strategy=resample_strategy)

        if rank == 0:
            total += 1
            particle_results = []
            for p in range(all_x.shape[0]):
                gen_text = tokenizer.decode(all_x[p, input_ids.shape[1]:].tolist()).split(tokenizer.eos_token)[0]
                pred_answer = extract_answer(gen_text)
                is_correct = check_math_answer(pred_answer, target_answer)
                p_logp = all_logp[p].sum().item()
                particle_results.append({
                    'particle': p, 'logp': p_logp,
                    'answer': pred_answer, 'correct': is_correct,
                })

            best_idx = max(range(len(particle_results)), key=lambda i: particle_results[i]['logp'])
            smc_correct = particle_results[best_idx]['correct']
            any_correct = any(pr['correct'] for pr in particle_results)

            correct_smc += smc_correct
            correct_any += any_correct

            results.append({
                'idx': idx, 'target': target_answer,
                'smc_correct': smc_correct, 'any_correct': any_correct,
                'particles': particle_results,
            })

            if total % 50 == 0:
                print(f"[{total}/{len(ds)}] SMC/BoN={correct_smc/total*100:.1f}% "
                      f"Pass@{total_particles}={correct_any/total*100:.1f}%")

        dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Dream-7B | N={total_particles} | {total} questions")
        print(f"{'='*60}")
        print(f"SR-SMC@{total_particles} / BoN@{total_particles}: {correct_smc/total*100:.2f}%")
        print(f"Pass@{total_particles}:    {correct_any/total*100:.2f}%")

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'{args.mode}_{total_particles}p.json'), 'w') as f:
            json.dump({
                'mode': args.mode, 'total_particles': total_particles,
                'smc_bon_accuracy': correct_smc / total * 100,
                'pass_at_n': correct_any / total * 100,
                'results': results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output_dir}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
