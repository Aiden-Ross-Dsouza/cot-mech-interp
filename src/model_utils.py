"""
src/model_utils.py
Model loading utilities for:
  - Gemma-2-2B-it  (main model, fp16) + Gemma Scope PLT transcoders
  - Gemma-2-9B-it  (paraphrase generator, 4-bit)
  - Llama-3.2-1B-Instruct (robustness check)
  - TransformerLens HookedTransformer (activation baselines)

All loading is lazy and cached: call get_main_model() multiple times safely.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.config import Config, ModelSpec

logger = logging.getLogger(__name__)

# ── Module-level caches (avoid reloading across script calls in the same process)
_main_model_cache: Optional[Tuple] = None
_paraphrase_model_cache: Optional[Tuple] = None
_lens_cache: Optional[object] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _torch_dtype(dtype_str: Optional[str]) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str is None:
        return torch.float32
    return mapping.get(dtype_str, torch.float16)


def _device(spec: ModelSpec) -> str:
    if spec.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU.")
        return "cpu"
    return spec.device


# ─────────────────────────────────────────────────────────────────────────────
# Main model: Gemma-2-2B-it + Gemma Scope PLT transcoders
# ─────────────────────────────────────────────────────────────────────────────

def load_main_model(cfg: Config) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Gemma-2-2B-it in fp16.

    Returns (model, tokenizer). The model is NOT wrapped with circuit-tracer
    here; circuit-tracer wraps it internally in graph_utils.py.
    """
    global _main_model_cache
    if _main_model_cache is not None:
        return _main_model_cache

    spec = cfg.models.main
    device = _device(spec)
    dtype = _torch_dtype(spec.dtype)

    logger.info(f"Loading {spec.name} ({spec.dtype}) on {device}…")
    tokenizer = AutoTokenizer.from_pretrained(spec.name)
    model = AutoModelForCausalLM.from_pretrained(
        spec.name,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    _main_model_cache = (model, tokenizer)
    logger.info(f"  ✓ {spec.name} loaded. "
                f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return _main_model_cache


def load_paraphrase_model(cfg: Config) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Gemma-2-9B-it in 4-bit (bitsandbytes) for paraphrase generation."""
    global _paraphrase_model_cache
    if _paraphrase_model_cache is not None:
        return _paraphrase_model_cache

    spec = cfg.models.paraphrase
    device = _device(spec)

    logger.info(f"Loading {spec.name} in 4-bit on {device}…")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.name)
    model = AutoModelForCausalLM.from_pretrained(
        spec.name,
        quantization_config=bnb_config,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    _paraphrase_model_cache = (model, tokenizer)
    logger.info(f"  ✓ {spec.name} (4-bit) loaded.")
    return _paraphrase_model_cache


def load_robustness_model(cfg: Config) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Llama-3.2-1B-Instruct in fp16 for cross-model robustness check."""
    spec = cfg.models.robustness
    device = _device(spec)
    dtype = _torch_dtype(spec.dtype)

    logger.info(f"Loading {spec.name} ({spec.dtype}) on {device}…")
    tokenizer = AutoTokenizer.from_pretrained(spec.name)
    model = AutoModelForCausalLM.from_pretrained(
        spec.name,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def load_lens(model_name: str, device: str = "cuda") -> object:
    """Load a TransformerLens HookedTransformer for activation extraction.

    Returns the HookedTransformer instance, cached after first call.
    Note: model_name must be supported by TransformerLens
    (e.g., 'gemma-2-2b' without '-it' suffix — TL handles the weights).
    """
    global _lens_cache
    if _lens_cache is not None:
        return _lens_cache

    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        raise ImportError(
            "transformer_lens not installed. Run: pip install transformer-lens"
        )

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    logger.info(f"Loading HookedTransformer for {model_name}…")
    lens_model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    ).to(device)
    lens_model.eval()
    _lens_cache = lens_model
    logger.info("  ✓ HookedTransformer loaded.")
    return _lens_cache


def unload_paraphrase_model() -> None:
    """Free the paraphrase model to reclaim VRAM for graph generation."""
    global _paraphrase_model_cache
    if _paraphrase_model_cache is not None:
        model, _ = _paraphrase_model_cache
        del model
        _paraphrase_model_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Paraphrase model freed from VRAM.")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate text with a loaded model
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
    seed: int = 42,
) -> str:
    """Greedy (default) or sampled generation. Returns decoded string only."""
    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()
