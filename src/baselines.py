"""
src/baselines.py
Five baseline measures for comparison against AGD:

1. activation_cosine   — layer-averaged residual-stream cosine at the answer position
2. kl_next_token       — KL divergence of the next-token distribution
3. cot_perplexity      — token-level perplexity of the CoT under the model
4. self_consistency_var — answer entropy across N sampled CoTs
5. random_feature_jaccard — null: Jaccard over randomly sampled features (not top-by-influence)

All functions accept a loaded model/tokenizer pair and return a scalar float,
or a pd.Series / dict for batch usage.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Activation-cosine baseline
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def activation_cosine(
    model,          # HookedTransformer (TransformerLens)
    prompt0: str,
    prompt1: str,
    layer_ids: Optional[List[int]] = None,
    device: str = "cuda",
) -> float:
    """Cosine similarity of residual-stream activations at the answer-token position.

    Computes per-layer cosine for each specified layer, then averages.
    The 'answer-token position' is the final token of each prompt.

    Parameters
    ----------
    model:
        A TransformerLens HookedTransformer instance.
    prompt0, prompt1:
        The two prompts to compare.
    layer_ids:
        List of layer indices. Defaults to all layers.
    device:
        Torch device string.

    Returns
    -------
    float in [0, 1] — average cosine similarity (mapped to [0,1]).
    """
    try:
        import transformer_lens
    except ImportError:
        raise ImportError("transformer_lens required for activation_cosine baseline.")

    def get_residuals(prompt: str) -> List[torch.Tensor]:
        """Return residual stream at the last token position for each layer."""
        _, cache = model.run_with_cache(
            prompt,
            names_filter=lambda name: name.endswith("hook_resid_post"),
            return_type=None,
        )
        # keys like 'blocks.0.hook_resid_post', shape: [1, seq_len, d_model]
        residuals = []
        keys = sorted(cache.keys(), key=lambda k: int(k.split(".")[1]))
        if layer_ids is not None:
            keys = [k for k in keys if int(k.split(".")[1]) in layer_ids]
        for k in keys:
            # Take the last token position
            residuals.append(cache[k][0, -1, :].float())
        return residuals

    res0 = get_residuals(prompt0)
    res1 = get_residuals(prompt1)

    cosines = []
    for r0, r1 in zip(res0, res1):
        c = F.cosine_similarity(r0.unsqueeze(0), r1.unsqueeze(0)).item()
        cosines.append((1.0 + c) / 2.0)  # map [-1,1] → [0,1]

    return float(np.mean(cosines)) if cosines else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. KL of next-token distribution
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def kl_next_token(
    model,
    tokenizer,
    prompt0: str,
    prompt1: str,
    temperature: float = 1.0,
) -> float:
    """KL divergence KL(P0 || P1) of next-token distributions.

    Parameters
    ----------
    temperature:
        Softmax temperature. Use 1.0 for raw logits comparison.

    Returns
    -------
    Non-negative float. Zero means identical distributions.
    """
    device = next(model.parameters()).device

    def get_log_probs(prompt: str) -> torch.Tensor:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        logits = model(**inputs).logits[0, -1, :]  # last position
        return F.log_softmax(logits / temperature, dim=-1)

    log_p0 = get_log_probs(prompt0)
    log_p1 = get_log_probs(prompt1)
    kl = F.kl_div(log_p1, log_p0.exp(), reduction="sum").item()
    return max(0.0, kl)  # numerical safety


# ─────────────────────────────────────────────────────────────────────────────
# 3. CoT perplexity
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def cot_perplexity(
    model,
    tokenizer,
    context: str,
    cot: str,
) -> float:
    """Token-level perplexity of the CoT string under the model.

    Computes cross-entropy loss over the CoT tokens conditioned on `context`.
    Lower perplexity = more likely CoT given the context.

    Returns
    -------
    float >= 1.0 (perplexity). Returns inf on empty CoT.
    """
    if not cot.strip():
        return float("inf")

    device = next(model.parameters()).device
    full_text = context + cot
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    context_len = len(tokenizer(context, add_special_tokens=False)["input_ids"])

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # Recompute loss restricted to CoT tokens only
    logits = outputs.logits[0, context_len - 1: -1, :]
    labels = inputs["input_ids"][0, context_len:]

    if labels.numel() == 0:
        return float("inf")

    loss = F.cross_entropy(logits, labels)
    return float(torch.exp(loss).item())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Self-consistency variance
# ─────────────────────────────────────────────────────────────────────────────

def self_consistency_variance(
    model,
    tokenizer,
    prompt: str,
    n: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42,
) -> float:
    """Answer entropy from N sampled CoTs: H(answers) / log2(|vocab_answers|).

    Extracts the final answer character/token from each sampled CoT,
    computes empirical entropy, and normalizes by log2(n).

    Returns
    -------
    float in [0, 1]. 0 = always same answer; 1 = maximally inconsistent.
    """
    from src.model_utils import generate_text

    torch.manual_seed(seed)
    device = next(model.parameters()).device

    answers = []
    for i in range(n):
        torch.manual_seed(seed + i)
        text = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            seed=seed + i,
        )
        # Heuristic: look for "Answer: X" or "The answer is X" or standalone (A)/(B)
        answer = _extract_answer_token(text)
        answers.append(answer)

    if not answers:
        return 0.0

    # Compute empirical entropy
    from collections import Counter
    counts = Counter(answers)
    total = sum(counts.values())
    probs = np.array([v / total for v in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = np.log2(n)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _extract_answer_token(text: str) -> str:
    """Heuristic extraction of the final answer token from a CoT string."""
    import re
    # Match patterns like "Answer: A", "the answer is (B)", "answer is B"
    for pattern in [
        r"[Aa]nswer[:\s]+\(?([A-E])\)?",
        r"therefore[,\s]+(?:the answer is\s+)?\(?([A-E])\)?",
        r"\(([A-E])\)\s*$",
        r"^([A-E])\.",
    ]:
        m = re.search(pattern, text)
        if m:
            return m.group(1).upper()
    # For numeric answers (GSM8K)
    m = re.search(r"####\s*(-?\d+)", text)
    if m:
        return m.group(1)
    # Fallback: last word
    words = text.strip().split()
    return words[-1] if words else "?"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Random-feature Jaccard (null baseline)
# ─────────────────────────────────────────────────────────────────────────────

def random_feature_jaccard(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    k: int = 64,
    seed: int = 42,
) -> float:
    """Jaccard similarity over *randomly sampled* k features (not top-by-influence).

    Used as a sanity null: if AGD's signal is purely from graph size or
    feature overlap due to model architecture (not mechanism), this baseline
    will also correlate.

    Returns
    -------
    float in [0, 1]. Expected ~0.5 for unrelated large graphs.
    """
    rng = random.Random(seed)

    all_ids0 = [n["feature_id"] for n in graph0["nodes"]]
    all_ids1 = [n["feature_id"] for n in graph1["nodes"]]

    sample0 = set(rng.sample(all_ids0, min(k, len(all_ids0))))
    sample1 = set(rng.sample(all_ids1, min(k, len(all_ids1))))

    intersection = len(sample0 & sample1)
    union = len(sample0 | sample1)
    return intersection / union if union > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Batch utility: compute all baselines for a pair (prompt0, prompt1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_baselines(
    model_hf,
    tokenizer,
    model_lens,
    graph0: Optional[Dict[str, Any]],
    graph1: Optional[Dict[str, Any]],
    prompt0: str,
    prompt1: str,
    cot0: str,
    k: int = 64,
    seed: int = 42,
    n_sc: int = 8,
    layer_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute all 5 baselines for a single pair. Returns a dict.

    Parameters
    ----------
    model_hf:      HuggingFace AutoModelForCausalLM (for KL, PPL, SC)
    tokenizer:     corresponding tokenizer
    model_lens:    TransformerLens HookedTransformer (for activation_cosine)
    graph0/1:      loaded graph dicts (for random_feature_jaccard)
    prompt0/1:     full prompts for the two conditions
    cot0:          CoT text of the base condition (for PPL)
    """
    results: Dict[str, float] = {}

    # 1. Activation-cosine
    try:
        results["activation_cosine"] = activation_cosine(
            model_lens, prompt0, prompt1, layer_ids=layer_ids
        )
    except Exception as e:
        logger.warning(f"activation_cosine failed: {e}")
        results["activation_cosine"] = float("nan")

    # 2. KL next-token
    try:
        results["kl_next_token"] = kl_next_token(model_hf, tokenizer, prompt0, prompt1)
    except Exception as e:
        logger.warning(f"kl_next_token failed: {e}")
        results["kl_next_token"] = float("nan")

    # 3. CoT perplexity
    try:
        results["cot_perplexity"] = cot_perplexity(model_hf, tokenizer, prompt0, cot0)
    except Exception as e:
        logger.warning(f"cot_perplexity failed: {e}")
        results["cot_perplexity"] = float("nan")

    # 4. Self-consistency variance (expensive: N=8 samples)
    try:
        results["sc_variance"] = self_consistency_variance(
            model_hf, tokenizer, prompt0, n=n_sc, seed=seed
        )
    except Exception as e:
        logger.warning(f"sc_variance failed: {e}")
        results["sc_variance"] = float("nan")

    # 5. Random-feature Jaccard null
    if graph0 is not None and graph1 is not None:
        try:
            results["random_jaccard"] = random_feature_jaccard(
                graph0, graph1, k=k, seed=seed
            )
        except Exception as e:
            logger.warning(f"random_jaccard failed: {e}")
            results["random_jaccard"] = float("nan")
    else:
        results["random_jaccard"] = float("nan")

    return results
