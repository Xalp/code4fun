"""
Distributed SMC evaluation on MATH-500 for Dream-7B, using lm-eval for scoring.
Launch: torchrun --nproc_per_node=8 eval_math_distributed.py [--mode smc|bon] [--local_particles 8]
"""
import argparse
import json
import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer
from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from generate_smc_distributed import generate_distributed_smc

# Use lm-eval's task for prompt formatting and scoring
from lm_eval import tasks as lm_tasks
from lm_eval.api.task import Task


def get_task_and_docs():
    """Load minerva_math500 task via lm-eval."""
    tm = lm_tasks.TaskManager()
    task_dict = lm_tasks.get_task_dict(['minerva_math500'], tm)
    task = list(task_dict.values())[0]
    docs = list(task.test_docs())
    return task, docs


def format_prompt(task, doc, tokenizer, n_shot=4):
    """Use lm-eval's fewshot + chat template to build prompt."""
    # Get fewshot context + question
    ctx = task.fewshot_context(doc, num_fewshot=n_shot)
    messages = [{"role": "user", "content": ctx}]
    chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return chat_text


def score_response(task, doc, response_text):
    """Use lm-eval's process_results to score a response."""
    # lm-eval expects the result in a specific format
    # For generate_until tasks, it's the generated string
    try:
        results = task.process_results(doc, [response_text])
        # results is a dict like {'exact_match': 0 or 1, 'math_verify': 0 or 1, ...}
        # Use math_verify if available, else exact_match
        if 'math_verify' in results:
            return results['math_verify'] > 0
        if 'exact_match' in results:
            return results['exact_match'] > 0
        return any(v > 0 for v in results.values() if isinstance(v, (int, float)))
    except Exception as e:
        print(f"Scoring error: {e}")
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
        print(f"Mode: {args.mode}, {world_size} GPUs x {args.local_particles} = {total_particles} particles")

    # Load model
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                        trust_remote_code=True, device_map={'': device}).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load task and docs via lm-eval
    task, docs = get_task_and_docs()
    if args.limit:
        docs = docs[:args.limit]

    results = []
    correct_smc = 0
    correct_any = 0
    total = 0

    for idx in tqdm(range(len(docs)), disable=(rank != 0)):
        doc = docs[idx]

        # Format prompt using lm-eval
        prompt_text = format_prompt(task, doc, tokenizer, args.n_shot)
        input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)
        attn_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

        # Distributed SMC generation
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
                gen_text = tokenizer.decode(all_x[p, input_ids.shape[1]:].tolist())
                # Truncate at eos
                gen_text = gen_text.split(tokenizer.eos_token)[0]
                # Also truncate at stop sequences if task defines them
                for stop in task.config.generation_kwargs.get('until', []):
                    if stop in gen_text:
                        gen_text = gen_text.split(stop)[0]

                p_logp = all_logp[p].sum().item()
                is_correct = score_response(task, doc, gen_text)
                particle_results.append({
                    'particle': p, 'logp': p_logp,
                    'correct': is_correct, 'text': gen_text[:300],
                })

            # SMC/BoN: pick particle with highest logp
            best_idx = max(range(len(particle_results)), key=lambda i: particle_results[i]['logp'])
            smc_correct = particle_results[best_idx]['correct']
            any_correct = any(pr['correct'] for pr in particle_results)

            correct_smc += smc_correct
            correct_any += any_correct

            results.append({
                'idx': idx, 'smc_correct': smc_correct,
                'any_correct': any_correct, 'particles': particle_results,
            })

            if total % 50 == 0:
                print(f"[{total}/{len(docs)}] SMC/BoN={correct_smc/total*100:.1f}% "
                      f"Pass@{total_particles}={correct_any/total*100:.1f}%")

        dist.barrier()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Dream-7B | N={total_particles} | {total} questions")
        print(f"{'='*60}")
        print(f"SR-SMC@{total_particles} / BoN@{total_particles}: {correct_smc/total*100:.2f}%")
        print(f"Pass@{total_particles}:    {correct_any/total*100:.2f}%")

        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, f'{args.mode}_{total_particles}p.json')
        with open(out_file, 'w') as f:
            json.dump({
                'mode': args.mode, 'total_particles': total_particles,
                'smc_bon_accuracy': correct_smc / total * 100,
                'pass_at_n': correct_any / total * 100,
                'results': results,
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved to {out_file}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
