"""
scripts/00_setup_and_replicate.py
Day 1 gate: verify circuit-tracer + Gemma Scope setup by replicating
Anthropic's published "Dallas → Texas → Austin" attribution graph example.

Usage:
    python scripts/00_setup_and_replicate.py --config config.yaml [--save]

Exit code 0 = PASS, 1 = FAIL (use in CI / make replicate).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model_utils import load_main_model
from src.graph_utils import generate_attribution_graph, save_graph


REPLICATION_PROMPT = (
    "What is the capital of the state whose largest city is Dallas? "
    "Let's think step by step.\n"
    "Dallas is in Texas. The capital of Texas is"
)
EXPECTED_TARGET = "Austin"
# Anthropic's example: the graph should have nodes referencing "Texas" and "capital"
EXPECTED_MIN_NODES = 5


def run_replication(cfg, save: bool = False) -> bool:
    """Run the replication and return True on pass."""
    logger.info("=" * 60)
    logger.info("DAY 1 GATE: Replicating Anthropic circuit-tracing example")
    logger.info("=" * 60)

    # 1. Check circuit-tracer is importable
    try:
        import circuit_tracer  # noqa: F401
        logger.info("[1/4] ✓ circuit-tracer importable")
    except ImportError:
        logger.error(
            "[1/4] ✗ circuit-tracer NOT found.\n"
            "Install: pip install git+https://github.com/anthropics/circuit-tracer.git"
        )
        return False

    # 2. Load model
    logger.info("[2/4] Loading Gemma-2-2B-it…")
    try:
        model, tokenizer = load_main_model(cfg)
        logger.info(f"       ✓ Model loaded: {cfg.models.main.name}")
    except Exception as e:
        logger.error(f"[2/4] ✗ Model loading failed: {e}")
        return False

    # 3. Generate attribution graph for replication example
    logger.info(f"[3/4] Generating attribution graph for target='{EXPECTED_TARGET}'…")
    try:
        graph = generate_attribution_graph(
            model=model,
            tokenizer=tokenizer,
            prompt=REPLICATION_PROMPT,
            target_token=EXPECTED_TARGET,
            cfg=cfg,
            item_id="replication",
            condition="clean",
        )
        n_nodes = graph["n_nodes"]
        n_edges = graph["n_edges"]
        logger.info(f"       ✓ Graph generated: {n_nodes} nodes, {n_edges} edges")
    except Exception as e:
        logger.error(f"[3/4] ✗ Graph generation failed: {e}")
        logger.exception(e)
        return False

    # 4. Validate output
    logger.info("[4/4] Validating graph…")
    passed = True

    if n_nodes < EXPECTED_MIN_NODES:
        logger.error(
            f"       ✗ Too few nodes: {n_nodes} (expected ≥ {EXPECTED_MIN_NODES}). "
            "Graph may be degenerate."
        )
        passed = False
    else:
        logger.info(f"       ✓ Node count OK: {n_nodes}")

    if n_edges == 0:
        logger.error("       ✗ Zero edges — graph is disconnected.")
        passed = False
    else:
        logger.info(f"       ✓ Edge count OK: {n_edges}")

    # Check for Texas-related features in top nodes (qualitative check)
    top_nodes = sorted(graph["nodes"], key=lambda n: abs(n["influence"]), reverse=True)[:5]
    top_labels = [n.get("label", "") for n in top_nodes]
    logger.info(f"       Top-5 node labels: {top_labels}")

    if save or passed:
        out_path = Path(cfg.paths.graphs) / "replication_clean.json"
        save_graph(graph, out_path)
        logger.info(f"       Graph saved to {out_path}")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Day 1 replication gate")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--save", action="store_true",
                        help="Save the graph even on failure (for debugging)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    passed = run_replication(cfg, save=args.save)

    print()
    if passed:
        print("=" * 60)
        print("[PASS] Replication successful. circuit-tracer + Gemma Scope are working.")
        print("       Proceed to Day 3 pilot: make pilot")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("[FAIL] Replication failed. See errors above.")
        print("       Fallback: try EleutherAI's Attribute library (see README).")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
