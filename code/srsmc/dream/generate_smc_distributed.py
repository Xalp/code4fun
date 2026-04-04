"""
Distributed SMC generation for Dream-7B.
8 GPUs × 8 particles = 64 particles with cross-GPU resampling.
"""
import torch
import torch.distributed as dist
import torch.distributions as dists
from torch.nn import functional as F


def sample_tokens(logits, temperature=1.0):
    """Sample tokens and return log-prob confidence."""
    if temperature > 0:
        logits = logits / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = dists.Categorical(logits=logits).sample()
            confidence = torch.gather(log_probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = log_probs.max(dim=-1)
    else:
        confidence, x0 = log_probs.max(dim=-1)
    return confidence, x0


@torch.inference_mode()
def generate_distributed_smc(model, prompt_ids, attention_mask=None,
                              steps=8, max_new_tokens=256, block_length=32,
                              temperature=1.0, threshold=0.9,
                              local_particles=8, resample_strategy="adaptive"):
    """
    Distributed SMC for Dream: each rank runs local_particles, resampling syncs across all ranks.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_particles = local_particles * world_size
    device = prompt_ids.device

    mask_token_id = model.config.mask_token_id
    eps = 1e-3
    prompt_length = prompt_ids.shape[1]
    max_length = prompt_length + max_new_tokens

    # Expand prompt for local particles
    x = F.pad(prompt_ids.repeat(local_particles, 1), (0, max_new_tokens), value=mask_token_id)

    num_blocks = max_new_tokens // block_length
    steps_per_block = steps // num_blocks
    timesteps = torch.linspace(1, eps, steps_per_block + 1, device=device)

    log_p = torch.zeros_like(x, dtype=torch.float32)
    log_w = torch.zeros(local_particles, dtype=torch.float32, device=device)

    if attention_mask is not None:
        attention_mask = attention_mask.repeat(local_particles, 1)
        attention_mask = F.pad(attention_mask, (0, max_new_tokens), value=1.0)

    for num_block in range(num_blocks):
        current_block_start = prompt_length + num_block * block_length
        current_block_end = current_block_start + block_length

        # Full forward (no cache for simplicity in distributed setting)
        logits = model(x).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        confidence, x0 = sample_tokens(logits, temperature=temperature)
        x[:, current_block_start] = x0[:, current_block_start]
        log_p[:, current_block_start] = confidence[:, current_block_start]
        log_w = log_w + confidence[:, current_block_start]

        i = 1
        while True:
            mask_index = (x == mask_token_id)
            mask_index[:, :current_block_start] = False
            mask_index[:, current_block_end:] = False

            # confidence_threshold algorithm
            logits = model(x).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            mask_logits = logits[mask_index]
            confidence, x0 = sample_tokens(mask_logits, temperature=temperature)

            full_confidence = torch.full_like(x[:, current_block_start:current_block_end],
                                              -torch.inf, dtype=logits.dtype, device=device)
            x_ = torch.zeros_like(x[:, current_block_start:current_block_end],
                                   device=device, dtype=torch.long) + mask_token_id
            local_mask = mask_index[:, current_block_start:current_block_end]
            x_[local_mask] = x0.clone()
            full_confidence[local_mask] = confidence

            current_transfer_tokens = local_mask.sum(dim=1)
            transfer_index = torch.zeros_like(x_, device=device, dtype=torch.bool)
            for j in range(local_particles):
                if current_transfer_tokens[j] == 0:
                    continue
                _, select_index = torch.topk(full_confidence[j], current_transfer_tokens[j])
                transfer_index[j, select_index] = True
                for k in range(1, current_transfer_tokens[j]):
                    if full_confidence[j, select_index[k]] < threshold:
                        transfer_index[j, select_index[k]] = False

            x[:, current_block_start:current_block_end][transfer_index] = x_[transfer_index]
            log_p[:, current_block_start:current_block_end][transfer_index] = full_confidence[transfer_index].float()
            log_w = log_w + torch.where(transfer_index, full_confidence, torch.zeros_like(full_confidence)).sum(dim=1).float()

            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break
            i += 1

        # --- Distributed resampling ---
        if resample_strategy == "never":
            continue

        all_log_w = [torch.zeros(local_particles, device=device) for _ in range(world_size)]
        dist.all_gather(all_log_w, log_w.contiguous())
        all_log_w = torch.cat(all_log_w)

        weights = torch.exp(all_log_w - all_log_w.max())
        weights = weights / weights.sum()
        ess = 1.0 / (weights.pow(2).sum())

        if ess < 0.5 * total_particles:
            all_x = [torch.zeros_like(x) for _ in range(world_size)]
            dist.all_gather(all_x, x.contiguous())
            all_x = torch.cat(all_x)

            all_logp = [torch.zeros_like(log_p) for _ in range(world_size)]
            dist.all_gather(all_logp, log_p.contiguous())
            all_logp = torch.cat(all_logp)

            k_idx = torch.zeros(total_particles, dtype=torch.long, device=device)
            if rank == 0:
                k_idx = torch.multinomial(weights, num_samples=total_particles, replacement=True)
                print(f"Distributed resampling at block {num_block}, ESS={ess:.2f}, "
                      f"unique={k_idx.unique().numel()}/{total_particles}")
            dist.broadcast(k_idx, src=0)

            all_x = all_x[k_idx]
            all_logp = all_logp[k_idx]
            x = all_x[rank * local_particles: (rank + 1) * local_particles].contiguous()
            log_p = all_logp[rank * local_particles: (rank + 1) * local_particles].contiguous()
            log_w.zero_()
        else:
            if rank == 0:
                print(f"Block {num_block}: ESS={ess:.2f}, no resampling")

    # Final gather
    all_x = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(all_x, x.contiguous())
    all_x = torch.cat(all_x)

    all_logp = [torch.zeros_like(log_p) for _ in range(world_size)]
    dist.all_gather(all_logp, log_p.contiguous())
    all_logp = torch.cat(all_logp)

    return all_x, all_logp
