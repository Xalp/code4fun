import accelerate
import torch
import transformers
from peft import prepare_model_for_kbit_training

from dllm.utils.configs import ModelArguments, TrainingArguments
from dllm.utils.utils import disable_caching_allocator_warmup, load_peft, print_main


def get_model(
    model_args,
    config: transformers.PretrainedConfig | None = None,
) -> transformers.PreTrainedModel:
    """
    Load a model with flexible input sources.

    Args:
        model_args: An optional dataclass or namespace containing model parameters.
        model_name_or_path: Optional direct model path or name (overrides model_args.model_name_or_path).
        dtype: Dtype (string or torch.dtype).
        load_in_4bit: Whether to load using 4-bit quantization (can override model_args.load_in_4bit).

    Returns:
        transformers.PreTrainedModel
    """
    model_name_or_path = getattr(model_args, "model_name_or_path")
    dtype = getattr(model_args, "dtype", "bfloat16")
    load_in_4bit = getattr(model_args, "load_in_4bit", False)
    attn_implementation = getattr(model_args, "attn_implementation", None)

    # Device map: skip when ZeRO-3
    device_map = (
        {"": accelerate.PartialState().local_process_index}
        if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        and torch.cuda.is_available()
        else None
    )

    quant_config = None
    if load_in_4bit and transformers.utils.is_bitsandbytes_available():
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load config and enable flash attention for LLaDA models
    if config is None:
        config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if hasattr(config, "flash_attention"):
        config.flash_attention = True

    params = {
        "dtype": dtype,
        "device_map": device_map,
        "quantization_config": quant_config,
        "attn_implementation": attn_implementation,
        "config": config,
        "trust_remote_code": True,
    }

    # Use local LLaDAModelLM for LLaDA models (supports kv-cache and flash attention)
    if hasattr(config, "flash_attention"):
        from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
        model = LLaDAModelLM.from_pretrained(model_name_or_path, **params)
    else:
        try:
            model = transformers.AutoModelForMaskedLM.from_pretrained(
                model_name_or_path, **params
            )
        except Exception:
            model = transformers.AutoModel.from_pretrained(model_name_or_path, **params)

    # --- if quantized, prepare for LoRA / QLoRA training ---
    if load_in_4bit and quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # Optionally train with lora
    model = load_peft(model, model_args)

    return model


def get_tokenizer(model_args) -> transformers.PreTrainedTokenizer:
    """
    Load a tokenizer with flexible input sources.

    Args:
        model_args: Optional dataclass or namespace containing model parameters.
        model: Optional model instance to configure tokenizer behavior.
        model_name_or_path: Optional direct model name or path (overrides model_args.model_name_or_path).

    Returns:
        transformers.PreTrainedTokenizer
    """
    # Lazy imports to avoid circular dependencies
    from transformers import (
        BertPreTrainedModel,
        ModernBertPreTrainedModel,
        RobertaPreTrainedModel,
    )

    from dllm.pipelines.a2d import (
        A2DLlamaLMHeadModel,
        A2DQwen2LMHeadModel,
        A2DQwen3LMHeadModel,
    )
    from dllm.pipelines.dream.models.modeling_dream import DreamModel
    from dllm.pipelines.llada2.models.modeling_llada2_moe import LLaDA2MoeModelLM
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM

    model_name_or_path = getattr(model_args, "model_name_or_path")

    # ---------------- Tokenizer loading ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
    )

    assert tokenizer.eos_token is not None or tokenizer.pad_token is not None

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        tokenizer.eos_token = tokenizer.pad_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.pad_token

    # If model is not provided, return as-is
    model_cfg = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Get model class, handling custom models that aren't in standard mapping
    config_class_name = type(model_cfg).__name__
    try:
        model_cls = transformers.AutoModel._model_mapping[type(model_cfg)]
    except KeyError:
        # Custom model not in standard mapping, set to None and check config name instead
        model_cls = None

    # ---------------- Model-specific customization ----------------
    if config_class_name == "LLaDAConfig" or (model_cls is not None and issubclass(model_cls, LLaDAModelLM)):
        tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        # tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token) # can not do this for llada base directly
        # TODO: for llada base, add special_tokens = {"<|start_header_id|>": 126346, "<|end_header_id|>": 126347, "<|eot_id|>": 126348}
        # fix bugs in chat template
        tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
    elif model_cls is not None and issubclass(model_cls, (LLaDAMoEModelLM, LLaDA2MoeModelLM)):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|role_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif model_cls is not None and issubclass(model_cls, DreamModel):
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif model_cls is not None and issubclass(
        model_cls,
        (BertPreTrainedModel, RobertaPreTrainedModel, ModernBertPreTrainedModel),
    ):
        tokenizer.eot_token = "[/Answer]"
        tokenizer.chat_template = """\
{% if messages[0]['role'] == 'system' %}
[SYS]
{{ messages[0]['content'] | trim }}
[/SYS]

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif -%}
{%- for message in loop_messages %}
{% if message['role'] == 'user' %}
[Question]
{{ message['content'] | trim }}
[/Question]

{% elif message['role'] == 'assistant' %}
[Answer]
{{ message['content'] | trim }}
[/Answer]

{% endif %}
{% endfor -%}
{%- if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
[Answer]
{% endif %}
"""
    elif model_cls is not None and issubclass(model_cls, A2DLlamaLMHeadModel):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif model_cls is not None and issubclass(model_cls, (A2DQwen2LMHeadModel, A2DQwen3LMHeadModel)):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    else:
        print_main("no tokenizer customization for model class:", model_cls)
    return tokenizer
