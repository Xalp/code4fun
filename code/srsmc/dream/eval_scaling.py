"""
Generate N independent samples per question with logp, score all, compute Pass@K/BoN@K/MajVote@K.
Uses lm-eval for prompt formatting and scoring. Data parallel across GPUs.

Launch: torchrun --nproc_per_node=8 eval_scaling.py --task minerva_math500 --total_samples 32
"""
import argparse
import json
import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer
from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from model.generation_utils_block import DreamGenerationMixin
import types

from lm_eval import tasks as lm_tasks


def generate_samples(model, tokenizer, input_ids, attn_mask, num_samples, batch_size=8,
                     max_new_tokens=256, steps=8, temperature=1.0, threshold=0.9):
    """Generate num_samples independent samples, returning all texts + logp."""
    all_texts = []
    all_logps = []
    prompt_len = input_ids.shape[1]

    for batch_start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - batch_start)
        # Expand input for this batch of particles
        batch_ids = input_ids.repeat(bs, 1)
        batch_mask = attn_mask.repeat(bs, 1) if attn_mask is not None else None

        with torch.inference_mode():
            output = model.diffusion_generate(
                batch_ids,
                attention_mask=batch_mask,
                max_new_tokens=max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=steps,
                temperature=temperature,
                alg='confidence_threshold',
                threshold=threshold,
                num_particles=bs,    # treat as independent particles
                resample_strategy="never",  # no resampling = independent
                block_length=32,
            )

        # output.sequences: [bs, seq_len]
        seqs = output.sequences
        for i in range(bs):
            gen_text = tokenizer.decode(seqs[i, prompt_len:].tolist())
            gen_text = gen_text.split(tokenizer.eos_token)[0]
            all_texts.append(gen_text)

        # Compute logp for each sample by re-scoring
        with torch.inference_mode():
            logits = model(seqs).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            # Gather log prob of actual tokens in generated region
            gen_tokens = seqs[:, prompt_len:]  # [bs, gen_len]
            gen_logp = torch.gather(log_probs[:, prompt_len:], -1,
                                     gen_tokens.unsqueeze(-1)).squeeze(-1)  # [bs, gen_len]
            # Mask out pad/mask tokens
            mask_token_id = model.config.mask_token_id
            valid = (gen_tokens != mask_token_id) & (gen_tokens != tokenizer.eos_token_id) & (gen_tokens != tokenizer.pad_token_id)
            for i in range(bs):
                lp = gen_logp[i][valid[i]].sum().item()
                all_logps.append(lp)

    return all_texts, all_logps


def score_response(task, doc, text):
    """Score a response using lm-eval's process_results."""
    try:
        # Truncate at stop sequences
        if hasattr(task, 'config') and hasattr(task.config, 'generation_kwargs'):
            for stop in task.config.generation_kwargs.get('until', []):
                if stop in text:
                    text = text.split(stop)[0]
        results = task.process_results(doc, [text])
        if 'math_verify' in results:
            return results['math_verify'] > 0, text
        if 'exact_match' in results:
            return results['exact_match'] > 0, text
        for k, v in results.items():
            if isinstance(v, (int, float)):
                return v > 0, text
        return False, text
    except Exception as e:
        return False, text


def compute_metrics(sample_results, K):
    """Compute Pass@K, BoN@K, MajVote@K from first K samples."""
    samples = sample_results[:K]
    # Pass@K
    pass_k = any(s['correct'] for s in samples)
    # BoN@K: pick highest logp
    best = max(samples, key=lambda s: s['logp'])
    bon_k = best['correct']
    # MajVote@K: majority vote on answers
    # Use the response text as the "answer" for voting
    answers = [(s['text'].strip(), s['correct']) for s in samples]
    vote_counts = Counter(a for a, _ in answers)
    majority_answer = vote_counts.most_common(1)[0][0]
    maj_correct = any(c for a, c in answers if a == majority_answer)
    return pass_k, bon_k, maj_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='minerva_math500', help='lm-eval task name')
    parser.add_argument('--total_samples', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8, help='particles per forward pass')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--n_shot', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./results_scaling')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    steps = args.max_new_tokens // 32

    if rank == 0:
        print(f"=== Dream-7B Scaling Eval ===")
        print(f"Task: {args.task}, {args.total_samples} samples/question, {world_size} GPUs")

    # Load model
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                        trust_remote_code=True, device_map={'': device}).eval()
    # Bind generation methods
    from model.generation_utils_smc_block import DreamGenerationMixin as SMCMixin
    model.diffusion_generate = types.MethodType(SMCMixin.diffusion_generate, model)
    model._sample = types.MethodType(SMCMixin._sample, model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load task via lm-eval
    tm = lm_tasks.TaskManager()
    task_dict = lm_tasks.get_task_dict([args.task], tm)
    task = list(task_dict.values())[0]
    docs = list(task.test_docs())
    if args.limit:
        docs = docs[:args.limit]

    # Partition across GPUs
    per_gpu = (len(docs) + world_size - 1) // world_size
    start = rank * per_gpu
    end = min(start + per_gpu, len(docs))
    my_docs = docs[start:end]
    my_indices = list(range(start, end))

    if rank == 0:
        print(f"Total docs: {len(docs)}, per GPU: ~{per_gpu}")

    local_results = []
    for local_idx, (global_idx, doc) in enumerate(zip(my_indices, my_docs)):
        # Format prompt
        ctx = task.fewshot_context(doc, num_fewshot=args.n_shot)
        messages = [{"role": "user", "content": ctx}]
        chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(chat_text, return_tensors='pt').input_ids.to(device)
        attn_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

        # Generate samples
        texts, logps = generate_samples(
            model, tokenizer, input_ids, attn_mask,
            num_samples=args.total_samples, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, steps=steps,
            temperature=args.temperature)

        # Score each sample
        sample_results = []
        for text, logp in zip(texts, logps):
            correct, cleaned = score_response(task, doc, text)
            sample_results.append({'text': cleaned[:300], 'logp': logp, 'correct': correct})

        # Compute metrics for K=8,16,32
        metrics = {}
        for K in [8, 16, 32]:
            if K <= args.total_samples:
                pass_k, bon_k, maj_k = compute_metrics(sample_results, K)
                metrics[K] = {'pass': pass_k, 'bon': bon_k, 'majvote': maj_k}

        local_results.append({
            'global_idx': global_idx,
            'metrics': metrics,
            'samples': sample_results,
        })

        if (local_idx + 1) % 20 == 0:
            # Quick stats
            for K in [8, 16, 32]:
                if K <= args.total_samples:
                    p = sum(r['metrics'][K]['pass'] for r in local_results) / len(local_results) * 100
                    b = sum(r['metrics'][K]['bon'] for r in local_results) / len(local_results) * 100
                    print(f"[GPU{rank}] {local_idx+1}/{len(my_docs)} K={K}: Pass={p:.1f}% BoN={b:.1f}%")

    # Gather all results on rank 0
    all_local = [None] * world_size
    dist.barrier()
    gathered = [None] * world_size
    # Use gloo for object gather
    import pickle
    local_bytes = pickle.dumps(local_results)
    local_tensor = torch.ByteTensor(list(local_bytes)).to(device)
    sizes = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(sizes, torch.tensor([len(local_bytes)], dtype=torch.long, device=device))

    if rank == 0:
        all_results = []
        # Receive from each rank
    max_size = max(s.item() for s in sizes)
    padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
    padded[:len(local_bytes)] = local_tensor
    all_padded = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(all_padded, padded)

    if rank == 0:
        all_results = []
        for i in range(world_size):
            data = bytes(all_padded[i][:sizes[i].item()].cpu().tolist())
            all_results.extend(pickle.loads(data))

        # Sort by global_idx
        all_results.sort(key=lambda r: r['global_idx'])

        print(f"\n{'='*70}")
        print(f"Dream-7B | {args.task} | {len(all_results)} questions | {args.total_samples} samples each")
        print(f"{'='*70}")
        header = f"{'K':<6}{'Pass@K':<12}{'BoN@K':<12}{'MajVote@K':<12}"
        print(header)
        print('-' * len(header))
        for K in [8, 16, 32]:
            if K <= args.total_samples:
                n = len(all_results)
                pass_k = sum(r['metrics'][K]['pass'] for r in all_results) / n * 100
                bon_k = sum(r['metrics'][K]['bon'] for r in all_results) / n * 100
                maj_k = sum(r['metrics'][K]['majvote'] for r in all_results) / n * 100
                print(f"{K:<6}{pass_k:<12.2f}{bon_k:<12.2f}{maj_k:<12.2f}")

        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, f'{args.task}_scaling_{args.total_samples}samples.json')
        # Save summary (without full text to keep file small)
        summary = []
        for r in all_results:
            summary.append({
                'global_idx': r['global_idx'],
                'metrics': r['metrics'],
                'sample_scores': [{'logp': s['logp'], 'correct': s['correct']} for s in r['samples']],
            })
        with open(out_file, 'w') as f:
            json.dump({
                'task': args.task,
                'total_samples': args.total_samples,
                'num_questions': len(all_results),
                'summary': {
                    K: {
                        'pass_at_k': sum(r['metrics'][K]['pass'] for r in all_results) / len(all_results) * 100,
                        'bon_at_k': sum(r['metrics'][K]['bon'] for r in all_results) / len(all_results) * 100,
                        'majvote_at_k': sum(r['metrics'][K]['majvote'] for r in all_results) / len(all_results) * 100,
                    } for K in [8, 16, 32] if K <= args.total_samples
                },
                'results': summary,
            }, f, indent=2)
        print(f"\nSaved to {out_file}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
