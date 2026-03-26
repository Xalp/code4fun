"""
MCMC / Gibbs sampling for masked diffusion language models.
After initial generation, iteratively re-mask random blocks and regenerate.
"""
import torch
import numpy as np
import torch.nn.functional as F
from generate import generate, get_num_transfer_tokens, get_transfer_index, add_gumbel_noise


@torch.inference_mode()
def generate_mcmc(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                  remasking='low_confidence', mask_id=126336, threshold=None, factor=None,
                  mcmc_steps=4):
    """
    Gibbs-style MCMC: generate initial sequence, then iteratively re-mask and regenerate blocks.
    Total compute ≈ (1 + mcmc_steps) * baseline, comparable to N-particle SMC when mcmc_steps = N-1.
    """
    # 1. Initial generation (baseline)
    x, nfe = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=block_length,
                      temperature=temperature, remasking=remasking, mask_id=mask_id,
                      threshold=threshold, factor=factor)

    prompt_length = prompt.shape[1]
    num_blocks_total = gen_length // block_length
    steps_per_block = steps // num_blocks_total

    # 2. Iterative refinement
    for r in range(mcmc_steps):
        # Pick a random block to refine
        block_idx = torch.randint(0, num_blocks_total, (1,)).item()
        block_start = prompt_length + block_idx * block_length
        block_end = block_start + block_length

        # Re-mask this block
        x[:, block_start:block_end] = mask_id

        # Re-generate this block
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        i = 0
        while True:
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, block_end:] = 0
            x0, transfer_index, _ = get_transfer_index(
                logits, temperature, remasking, mask_index, x,
                num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[transfer_index] = x0[transfer_index]
            i += 1

        print(f"MCMC step {r+1}/{mcmc_steps}: refined block {block_idx}")

    return x, nfe
