# SMC Ablation Study Report: Quantifying "Overtake" Events

## Objective
To quantify the effectiveness of Sequential Monte Carlo (SMC) resampling in recovering from locally sub-optimal choices during text generation. specifically, we want to measure how often a "non-dominant" path (one with lower probability at the start of a block) eventually surpasses the "dominant" path to become the winner by the end of the block.

## Methodology
*   **Task**: `minerva_math500` (Subset of MATH dataset).
*   **Model**: `LLaDA-1.5` (8B).
*   **Configuration**: 
    *   4-shot evaluation.
    *   Generation: `length=256`, `block_length=32`, `steps=8`, `num_particles=4`.
*   **Metric**: **Overtake Percentage**.
    *   We track the set of "Dominant Particles" (those with the highest probability) at the **Start** of each block (post-resampling from the previous block).
    *   We identify the "Winning Particle" (highest probability) at the **End** of the block.
    *   An **Overtake Event** occurs if the Winning Particle at the end was *not* in the Dominant Set at the start.

## Results

### Performance
*   **Accuracy (math_verify)**: 41.8% (Pass@1)

### SMC Dynamics
*   **Total Blocks Analyzed**: 4032
*   **Overtake Events**: **1159 (28.75%)**

## Conclusion
The data shows that **28.75%** of the time, the sequence that appeared "best" at the beginning of a generation block was eventually outperformed by a different candidate. 

This confirms that SMC's resampling mechanism plays a critical active role: it keeps alternative (initially lower-probability) paths alive long enough for them to demonstrate their superiority, effectively correcting local greedy errors that standard decoding methods might typically lock into.
