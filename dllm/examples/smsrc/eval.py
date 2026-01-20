from types import SimpleNamespace
from dataclasses import dataclass

import accelerate
import torch
import torch.nn.functional as F
from tqdm import tqdm
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype

import dllm
from dllm.pipelines.llada.eval import LLaDAEvalConfig, LLaDAEvalHarness
# Import our local SMC generator
try:
    from .generate_smc import generate_with_prefix_cache_smc
except ImportError:
    from generate_smc import generate_with_prefix_cache_smc

@register_model("llada_smc")
class LLaDASMC_EvalHarness(LLaDAEvalHarness):
    def __init__(
        self,
        config: LLaDAEvalConfig | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        # Add any SMC specific config if needed, e.g. num_particles
        self.num_particles = kwargs.get("num_particles", 4)
        if hasattr(config, "num_particles"):
            self.num_particles = config.num_particles

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate greedily until a stopping sequence, using SMC"""
        out = []
        # We don't use MDLMSampler here, we use our SMC generator

        for instance in tqdm(requests, desc="Generating (SMC)..."):
            context, gen_kwargs = instance.args  # type: ignore
            prompt_ids = self.tokenizer(context)["input_ids"]
            prompt = torch.tensor(prompt_ids, device=self.device, dtype=torch.long).unsqueeze(0)
            
            stop_tokens = gen_kwargs["until"]
            
            # Call SMC generator
            # Note: generate_with_prefix_cache_smc returns (x, nfe)
            # x shape: (num_particles * batch, len) -> (1, len) because it selects best
            
            # Although the function signature of generate_with_prefix_cache_smc supports batching via prompt.shape[0],
            # the eval loop processes one instance at a time (unless we optimized it, but let's stick to safe loop).
            # The original eval.py processes in a loop but with batch=1 for prompt usually.
            
            generated_ids, _ = generate_with_prefix_cache_smc(
                model=self.model,
                prompt=prompt,
                steps=self.steps,
                gen_length=self.max_new_tokens,
                block_length=self.block_size,
                temperature=0.0, # Usually 0 for deterministic-ish sampling, but SMC adds gumbel noise anyway
                remasking=self.remasking,
                num_particles=int(self.num_particles),
                # pass other args if needed
            )
            
            # generated_ids is (1, total_len) because generate_smc returns x[idx:idx+1] (best particle)
            
            generated_answer = self.tokenizer.decode(
                generated_ids[0][prompt.shape[1] :], skip_special_tokens=False
            )
            
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            out.append(generated_answer)
            
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

if __name__ == "__main__":
    cli_evaluate()
