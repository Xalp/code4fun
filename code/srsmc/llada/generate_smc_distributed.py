"""
Distributed SMC generation for LLaDA.
8 GPUs × 8 particles = 64 particles with cross-GPU resampling.
Launch with: torchrun --nproc_per_node=8 eval_math_distributed.py
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from generate_smc import add_gumbel_noise, get_num_transfer_tokens, get_transfer_index, _compute_weight


@torch.inference_mode()
def generate_distributed_smc(model, prompt, steps=8, gen_length=256, block_length=32,
                              temperature=1.0, remasking='low_confidence', mask_id=126336,
                              threshold=0.9, local_particles=8, resample_strategy="adaptive"):
    """
    Distributed SMC: each rank runs local_particles, resampling syncs across all ranks.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_particles = local_particles * world_size

    x = torch.full((local_particles, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_length = prompt.shape[1]

    logp = torch.zeros_like(x, dtype=torch.float32)
    log_w = torch.zeros(local_particles, device=model.device)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        idx_s = prompt_length + num_block * block_length
        idx_t = idx_s + block_length
        block_mask_index = (x[:, idx_s:idx_t] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        i = 0

        while True:
            if (x[:, idx_s:idx_t] == mask_id).sum() == 0:
                break
            mask_index = (x == mask_id)
            logits = model(x).logits
            nfe += 1
            mask_index[:, idx_t:] = 0
            x0, transfer_index, x0_logp = get_transfer_index(
                logits, temperature, remasking, mask_index, x,
                num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[transfer_index] = x0[transfer_index]
            logp[transfer_index] = x0_logp[transfer_index]
            log_w = log_w + _compute_weight(logits, x0_logp, transfer_index, "confidence")
            i += 1

        # --- Distributed resampling ---
        if resample_strategy == "never":
            continue  # BoN: no resampling

        # All-gather log_w from all ranks → [total_particles]
        all_log_w = [torch.zeros(local_particles, device=model.device) for _ in range(world_size)]
        dist.all_gather(all_log_w, log_w)
        all_log_w = torch.cat(all_log_w)  # [64]

        weights = torch.exp(all_log_w - all_log_w.max())
        weights = weights / weights.sum()
        ess = 1.0 / (weights.pow(2).sum())

        if ess < 0.5 * total_particles:
            # All-gather x and logp
            all_x = [torch.zeros_like(x) for _ in range(world_size)]
            dist.all_gather(all_x, x.contiguous())
            all_x = torch.cat(all_x)  # [64, seq_len]

            all_logp = [torch.zeros_like(logp) for _ in range(world_size)]
            dist.all_gather(all_logp, logp.contiguous())
            all_logp = torch.cat(all_logp)

            # Resample on rank 0, broadcast indices
            k_idx = torch.zeros(total_particles, dtype=torch.long, device=model.device)
            if rank == 0:
                k_idx = torch.multinomial(weights, num_samples=total_particles, replacement=True)
                print(f"Distributed resampling at block {num_block}, ESS={ess:.2f}, "
                      f"unique particles: {k_idx.unique().numel()}/{total_particles}")
            dist.broadcast(k_idx, src=0)

            all_x = all_x[k_idx]
            all_logp = all_logp[k_idx]

            # Each rank takes its slice
            x = all_x[rank * local_particles : (rank + 1) * local_particles].contiguous()
            logp = all_logp[rank * local_particles : (rank + 1) * local_particles].contiguous()
            log_w.zero_()
        else:
            if rank == 0:
                print(f"Block {num_block}: ESS={ess:.2f}, no resampling")

    # Final: gather all, return all particles + logp for flexibility
    all_x = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(all_x, x.contiguous())
    all_x = torch.cat(all_x)

    all_logp = [torch.zeros_like(logp) for _ in range(world_size)]
    dist.all_gather(all_logp, logp.contiguous())
    all_logp = torch.cat(all_logp)

    return all_x, all_logp, nfe
