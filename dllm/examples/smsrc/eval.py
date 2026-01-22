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
from dllm.pipelines.dream.eval import DreamEvalConfig, DreamEvalHarness

# Import our local SMC generator
try:
    from .generate_smc import generate_with_prefix_cache_smc
except ImportError:
    pass # Will be imported in methods if needed or rely on script execution context

@register_model("llada_smc")
class LLaDASMC_EvalHarness(LLaDAEvalHarness):
    def __init__(
        self,
        config: LLaDAEvalConfig | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.num_particles = int(kwargs.get("num_particles", 4))
        self.threshold = kwargs.get("threshold", None)
        self.factor = kwargs.get("factor", None)
        self.use_smc = kwargs.get("use_smc", True)

        if config:
            if hasattr(config, "num_particles"): self.num_particles = config.num_particles
            if hasattr(config, "threshold"): self.threshold = config.threshold
            if hasattr(config, "factor"): self.factor = config.factor
            if hasattr(config, "use_smc"): self.use_smc = config.use_smc
        
        if str(self.use_smc).lower() == "false":
            self.num_particles = 1
        
        if self.threshold is not None:
            self.threshold = float(self.threshold)
        if self.factor is not None:
             self.factor = float(self.factor)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate greedily until a stopping sequence, using SMC"""
        # Ensure generate_smc is imported
        try:
            from .generate_smc import generate_with_prefix_cache_smc
        except ImportError:
            from generate_smc import generate_with_prefix_cache_smc

        out = []
        for instance in tqdm(requests, desc="Generating (SMC-LLaDA)..."):
            context, gen_kwargs = instance.args 
            
            # Simple batching implementation (batch size 1 for safety with SMC)
            prompt_ids = self.tokenizer(context)["input_ids"]
            prompt = torch.tensor(prompt_ids, device=self.device, dtype=torch.long).unsqueeze(0)
            
            stop_tokens = gen_kwargs["until"]
            max_gen = self.max_new_tokens
            
            generated_ids, _ = generate_with_prefix_cache_smc(
                model=self.model,
                prompt=prompt,
                steps=self.steps,
                gen_length=max_gen,
                block_length=self.block_size,
                temperature=self.temperature, 
                remasking=self.remasking,
                num_particles=self.num_particles,
                threshold=self.threshold,
                factor=self.factor,
                mask_id=126336 # LLaDA default
            )
            
            # generated_ids is (1, total_len) because generate_smc returns best particle
            generated_answer = self.tokenizer.decode(
                generated_ids[0][prompt.shape[1] :], skip_special_tokens=False
            )
            
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            out.append(generated_answer)
            
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

@register_model("dream_smc")
class DreamSMC_EvalHarness(DreamEvalHarness):
    def __init__(
        self,
        config: DreamEvalConfig | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.num_particles = int(kwargs.get("num_particles", 4))
        self.threshold = kwargs.get("threshold", None)
        self.factor = kwargs.get("factor", None)
        self.use_smc = kwargs.get("use_smc", True)

        if config:
            if hasattr(config, "num_particles"): self.num_particles = config.num_particles
            if hasattr(config, "threshold"): self.threshold = config.threshold
            if hasattr(config, "factor"): self.factor = config.factor
            if hasattr(config, "use_smc"): self.use_smc = config.use_smc
        
        if str(self.use_smc).lower() == "false":
            self.num_particles = 1

        if self.threshold is not None:
            self.threshold = float(self.threshold)
        if self.factor is not None:
             self.factor = float(self.factor)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate greedily until a stopping sequence, using SMC for Dream"""
        try:
            from .generate_smc import generate_with_prefix_cache_smc
        except ImportError:
            from generate_smc import generate_with_prefix_cache_smc

        out = []
        for instance in tqdm(requests, desc="Generating (SMC-Dream)..."):
            context, gen_kwargs = instance.args
            
            prompts = [context]
            if self.add_bos_token:
                prompts = [self.tokenizer.bos_token + p for p in prompts]
            
            prompt_ids_list = [
                self.tokenizer(p, return_tensors="pt", padding=False).input_ids.squeeze().to(self.device)
                for p in prompts
            ]
            # Batch size 1 logic
            prompt = prompt_ids_list[0].unsqueeze(0)
            
            # Handle truncation if needed (simplified from DreamEvalHarness)
            if prompt.shape[1] > self.max_length - self.max_new_tokens:
                 cutoff_len = self.max_length - self.max_new_tokens
                 prompt = prompt[:, -cutoff_len:]

            stop_tokens = gen_kwargs["until"]
            
            # Dream typically does full generation (diffusion steps). 
            # We assume block_length = gen_length for standard diffusion unless specified.
            # But generate_smc supports AR. We'll default to using 'steps' and 'max_new_tokens'.
            
            generated_ids, _ = generate_with_prefix_cache_smc(
                model=self.model,
                prompt=prompt,
                steps=self.steps,
                gen_length=self.max_new_tokens,
                block_length=self.max_new_tokens, # Defaulting to full block for Dream
                temperature=self.temperature, 
                remasking='low_confidence', # Dream default usually? Or from config?
                num_particles=self.num_particles,
                threshold=self.threshold,
                factor=self.factor,
                mask_id=self.tokenizer.mask_token_id
            )
            
            responses = [
                g.removeprefix("<|endoftext|>").split(self.tokenizer.eos_token, 1)[0]
                for g in self.tokenizer.batch_decode(generated_ids)
            ]
            
            r = responses[0]
            if not self.escape_until:
                for s in stop_tokens:
                    r = r.split(s)[0]
            
            out.append(r)
            
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
                
        return out


if __name__ == "__main__":
    cli_evaluate()
