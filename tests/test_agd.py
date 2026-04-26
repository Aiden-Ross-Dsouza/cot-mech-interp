"""
tests/test_agd.py
Unit tests for the AGD metric (src/agd.py).

These tests are model-free: they use hand-crafted synthetic graph dicts
to verify the mathematical properties of AGD components.

Run with: pytest tests/test_agd.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agd import (
    weighted_jaccard,
    edge_cosine,
    compute_agd,
    agd_alpha_sweep,
    agd_k_sweep,
    agd_layer_band,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic graph fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_graph(node_ids_weights: dict, edge_list: list, layer_map: dict = None) -> dict:
    """Create a minimal synthetic graph dict."""
    nodes = [
        {
            "feature_id": fid,
            "layer": layer_map.get(fid, 0) if layer_map else 0,
            "influence": w,
            "label": f"feat_{fid}",
        }
        for fid, w in node_ids_weights.items()
    ]
    edges = [
        {"src": src, "dst": dst, "weight": w}
        for src, dst, w in edge_list
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "item_id": "test",
        "condition": "test",
        "pruning_threshold": 0.8,
    }


# Identical graphs: same features and edges
GRAPH_A = make_graph(
    {"f1": 0.9, "f2": 0.7, "f3": 0.5, "f4": 0.3},
    [("f1", "f2", 0.8), ("f2", "f3", 0.6), ("f3", "f4", 0.4)],
)
GRAPH_A_COPY = make_graph(
    {"f1": 0.9, "f2": 0.7, "f3": 0.5, "f4": 0.3},
    [("f1", "f2", 0.8), ("f2", "f3", 0.6), ("f3", "f4", 0.4)],
)

# Completely disjoint from GRAPH_A
GRAPH_B = make_graph(
    {"g1": 0.9, "g2": 0.7, "g3": 0.5, "g4": 0.3},
    [("g1", "g2", 0.8), ("g2", "g3", 0.6), ("g3", "g4", 0.4)],
)

# Partial overlap with GRAPH_A (shares f1, f2)
GRAPH_C = make_graph(
    {"f1": 0.8, "f2": 0.6, "g1": 0.5, "g2": 0.3},
    [("f1", "f2", 0.7), ("g1", "g2", 0.5)],
)

# Layer-annotated graph
GRAPH_LAYER = make_graph(
    {"e1": 0.9, "m1": 0.7, "l1": 0.5},
    [("e1", "m1", 0.8), ("m1", "l1", 0.6)],
    layer_map={"e1": 2, "m1": 12, "l1": 20},
)
GRAPH_LAYER_COPY = make_graph(
    {"e1": 0.9, "m1": 0.7, "l1": 0.5},
    [("e1", "m1", 0.8), ("m1", "l1", 0.6)],
    layer_map={"e1": 2, "m1": 12, "l1": 20},
)


# ─────────────────────────────────────────────────────────────────────────────
# weighted_jaccard tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedJaccard:
    def test_identical_sets_returns_one(self):
        N = {"f1": 0.9, "f2": 0.7, "f3": 0.5}
        assert weighted_jaccard(N, N) == pytest.approx(1.0)

    def test_disjoint_sets_returns_zero(self):
        N0 = {"f1": 0.9, "f2": 0.7}
        N1 = {"g1": 0.8, "g2": 0.6}
        assert weighted_jaccard(N0, N1) == pytest.approx(0.0)

    def test_partial_overlap(self):
        N0 = {"f1": 1.0, "f2": 1.0}  # sum = 2
        N1 = {"f1": 1.0, "g1": 1.0}  # shares f1
        # intersection: min(1,1)=1 for f1; union: max(1,1)+max(1,0)+max(0,1) = 1+1+1=3
        result = weighted_jaccard(N0, N1)
        assert result == pytest.approx(1.0 / 3.0)

    def test_range_always_zero_to_one(self):
        """J_w must be in [0, 1] for arbitrary inputs."""
        import random
        rng = random.Random(42)
        for _ in range(100):
            n_feats = rng.randint(1, 20)
            N0 = {f"f{i}": rng.random() for i in range(n_feats)}
            N1 = {f"f{rng.randint(0, n_feats)}" : rng.random() for _ in range(rng.randint(1, n_feats))}
            jw = weighted_jaccard(N0, N1)
            assert 0.0 <= jw <= 1.0 + 1e-9, f"J_w out of range: {jw}"

    def test_empty_both_returns_one(self):
        assert weighted_jaccard({}, {}) == pytest.approx(1.0)

    def test_one_empty_returns_zero(self):
        N = {"f1": 0.5}
        assert weighted_jaccard(N, {}) == pytest.approx(0.0)
        assert weighted_jaccard({}, N) == pytest.approx(0.0)

    def test_scale_invariant(self):
        """J_w should change if we scale one set (min/max change)."""
        N0 = {"f1": 1.0, "f2": 1.0}
        N1 = {"f1": 2.0, "f2": 2.0}
        # min(1,2)+min(1,2) / max(1,2)+max(1,2) = 2/4 = 0.5
        assert weighted_jaccard(N0, N1) == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# edge_cosine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCosine:
    def test_identical_vectors_returns_one(self):
        E = np.array([0.8, 0.6, 0.4], dtype=np.float32)
        assert edge_cosine(E, E) == pytest.approx(1.0)

    def test_zero_both_returns_one(self):
        E = np.zeros(4, dtype=np.float32)
        assert edge_cosine(E, E) == pytest.approx(1.0)

    def test_zero_one_returns_zero(self):
        E0 = np.zeros(4, dtype=np.float32)
        E1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        assert edge_cosine(E0, E1) == pytest.approx(0.0)

    def test_opposite_vectors_returns_zero(self):
        # cos(-1) → (1 + -1)/2 = 0
        E0 = np.array([1.0, 0.0], dtype=np.float32)
        E1 = np.array([-1.0, 0.0], dtype=np.float32)
        assert edge_cosine(E0, E1) == pytest.approx(0.0)

    def test_range_zero_to_one(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            E0 = rng.standard_normal(20).astype(np.float32)
            E1 = rng.standard_normal(20).astype(np.float32)
            se = edge_cosine(E0, E1)
            assert 0.0 <= se <= 1.0 + 1e-6, f"S_e out of range: {se}"


# ─────────────────────────────────────────────────────────────────────────────
# compute_agd tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAGD:
    def test_identical_graphs_agd_zero(self):
        result = compute_agd(GRAPH_A, GRAPH_A_COPY, alpha=0.5, k=4)
        assert result["agd"] == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_graphs_agd_one(self):
        result = compute_agd(GRAPH_A, GRAPH_B, alpha=0.5, k=4)
        assert result["agd"] == pytest.approx(1.0, abs=1e-3)

    def test_partial_overlap_between_zero_and_one(self):
        result = compute_agd(GRAPH_A, GRAPH_C, alpha=0.5, k=4)
        assert 0.0 < result["agd"] < 1.0

    def test_agd_range(self):
        """AGD must always be in [0, 1]."""
        import random
        rng = random.Random(99)

        def rand_graph(n_nodes=6, n_edges=5):
            nodes = {f"f{i}": rng.random() for i in range(n_nodes)}
            edges = [(f"f{rng.randint(0,n_nodes-1)}", f"f{rng.randint(0,n_nodes-1)}", rng.random())
                     for _ in range(n_edges)]
            return make_graph(nodes, edges)

        for _ in range(30):
            g0 = rand_graph()
            g1 = rand_graph()
            r = compute_agd(g0, g1)
            assert 0.0 <= r["agd"] <= 1.0 + 1e-6, f"AGD={r['agd']} out of [0,1]"

    def test_alpha_zero_uses_only_edge_term(self):
        """With alpha=0, AGD = 1 - S_e only."""
        r = compute_agd(GRAPH_A, GRAPH_A_COPY, alpha=0.0, k=4)
        assert r["agd"] == pytest.approx(1.0 - r["se"], abs=1e-6)

    def test_alpha_one_uses_only_node_term(self):
        """With alpha=1, AGD = 1 - J_w only."""
        r = compute_agd(GRAPH_A, GRAPH_A_COPY, alpha=1.0, k=4)
        assert r["agd"] == pytest.approx(1.0 - r["jw"], abs=1e-6)

    def test_returns_expected_keys(self):
        r = compute_agd(GRAPH_A, GRAPH_B)
        assert set(r.keys()) == {"agd", "jw", "se", "n0", "n1"}

    def test_n0_n1_correct(self):
        r = compute_agd(GRAPH_A, GRAPH_B)
        assert r["n0"] == GRAPH_A["n_nodes"]
        assert r["n1"] == GRAPH_B["n_nodes"]


# ─────────────────────────────────────────────────────────────────────────────
# agd_alpha_sweep tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaSweep:
    def test_sweep_returns_all_alphas(self):
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = agd_alpha_sweep(GRAPH_A, GRAPH_B, alpha_grid=alphas, k=4)
        assert set(result.keys()) == set(alphas)

    def test_identical_graphs_always_zero(self):
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = agd_alpha_sweep(GRAPH_A, GRAPH_A_COPY, alpha_grid=alphas, k=4)
        for alpha, agd_val in result.items():
            assert agd_val == pytest.approx(0.0, abs=1e-5), \
                f"Expected 0 for alpha={alpha}, got {agd_val}"

    def test_monotonicity_not_required_but_bounded(self):
        """All sweep values must be in [0,1]."""
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = agd_alpha_sweep(GRAPH_A, GRAPH_C, alpha_grid=alphas, k=4)
        for a, v in result.items():
            assert 0.0 <= v <= 1.0 + 1e-6, f"AGD={v} out of range for alpha={a}"


# ─────────────────────────────────────────────────────────────────────────────
# agd_layer_band tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerBand:
    def test_full_range_equals_all_nodes(self):
        """Including all layers should match compute_agd on the full graph."""
        full_agd = compute_agd(GRAPH_LAYER, GRAPH_LAYER_COPY)["agd"]
        band_agd = agd_layer_band(GRAPH_LAYER, GRAPH_LAYER_COPY, (0, 25))
        assert band_agd == pytest.approx(full_agd, abs=1e-5)

    def test_empty_band_returns_valid_value(self):
        """A layer band with no matching nodes should not crash."""
        # e1 is at layer 2, m1 at 12, l1 at 20. Band [30, 40] → empty.
        result = agd_layer_band(GRAPH_LAYER, GRAPH_LAYER_COPY, (30, 40))
        assert 0.0 <= result <= 1.0 + 1e-6

    def test_band_subsets_nodes(self):
        """Early-band AGD (layers 0–8) uses only e1 (layer=2)."""
        result = agd_layer_band(GRAPH_LAYER, GRAPH_LAYER_COPY, (0, 8))
        # Both graphs have identical e1 features → should be ~0
        assert result == pytest.approx(0.0, abs=1e-5)
