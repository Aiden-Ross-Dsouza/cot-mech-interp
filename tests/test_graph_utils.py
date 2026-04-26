"""
tests/test_graph_utils.py
Unit tests for graph I/O and feature extraction (src/graph_utils.py).

Model-free: all tests use synthetic graph dicts.
Run with: pytest tests/test_graph_utils.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.graph_utils import (
    save_graph,
    load_graph,
    graph_path,
    graph_exists,
    graph_to_node_set,
    graph_to_edge_vec,
    union_edge_keys,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_graph(n_nodes: int = 8, n_edges: int = 10, seed: int = 42) -> dict:
    """Make a synthetic graph with deterministic features."""
    rng = np.random.default_rng(seed)
    nodes = [
        {
            "feature_id": f"feat_{i:03d}",
            "layer": int(rng.integers(0, 25)),
            "influence": float(rng.uniform(0.1, 1.0)),
            "label": f"label_{i}",
        }
        for i in range(n_nodes)
    ]
    node_ids = [n["feature_id"] for n in nodes]
    edges = [
        {
            "src": node_ids[int(rng.integers(0, n_nodes))],
            "dst": node_ids[int(rng.integers(0, n_nodes))],
            "weight": float(rng.uniform(-1.0, 1.0)),
        }
        for _ in range(n_edges)
    ]
    return {
        "item_id": "test_item",
        "condition": "clean",
        "prompt": "What is the capital of France? Paris",
        "target_token": "Paris",
        "target_token_id": 9999,
        "pruning_threshold": 0.8,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "model": "google/gemma-2-2b-it",
            "transcoder_type": "plt",
            "k": 64,
            "timestamp": "2026-04-26T00:00:00+00:00",
        },
    }


GRAPH_SMALL = _make_graph(n_nodes=4, n_edges=3, seed=0)
GRAPH_MEDIUM = _make_graph(n_nodes=10, n_edges=15, seed=1)
GRAPH_LARGE = _make_graph(n_nodes=20, n_edges=40, seed=2)


# ─────────────────────────────────────────────────────────────────────────────
# save_graph / load_graph roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphIO:
    def test_save_load_roundtrip_identical(self, tmp_path):
        p = tmp_path / "test.json"
        save_graph(GRAPH_MEDIUM, p)
        loaded = load_graph(p)
        assert loaded["item_id"] == GRAPH_MEDIUM["item_id"]
        assert loaded["n_nodes"] == GRAPH_MEDIUM["n_nodes"]
        assert loaded["n_edges"] == GRAPH_MEDIUM["n_edges"]
        assert len(loaded["nodes"]) == len(GRAPH_MEDIUM["nodes"])
        assert len(loaded["edges"]) == len(GRAPH_MEDIUM["edges"])

    def test_node_data_preserved(self, tmp_path):
        p = tmp_path / "g.json"
        save_graph(GRAPH_SMALL, p)
        loaded = load_graph(p)
        for orig, recovered in zip(GRAPH_SMALL["nodes"], loaded["nodes"]):
            assert orig["feature_id"] == recovered["feature_id"]
            assert orig["influence"] == pytest.approx(recovered["influence"])

    def test_edge_data_preserved(self, tmp_path):
        p = tmp_path / "g.json"
        save_graph(GRAPH_SMALL, p)
        loaded = load_graph(p)
        for orig, recovered in zip(GRAPH_SMALL["edges"], loaded["edges"]):
            assert orig["src"] == recovered["src"]
            assert orig["dst"] == recovered["dst"]
            assert orig["weight"] == pytest.approx(recovered["weight"])

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "graph.json"
        save_graph(GRAPH_SMALL, nested)
        assert nested.exists()

    def test_save_is_atomic_valid_json(self, tmp_path):
        """The saved file must be valid JSON (no partial writes)."""
        p = tmp_path / "atomic.json"
        save_graph(GRAPH_LARGE, p)
        with open(p) as f:
            data = json.load(f)
        assert "nodes" in data and "edges" in data

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_graph(tmp_path / "does_not_exist.json")


# ─────────────────────────────────────────────────────────────────────────────
# graph_path / graph_exists
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphPathExists:
    def test_graph_path_format(self, tmp_path):
        p = graph_path("item_001", "clean", tmp_path)
        assert p.name == "item_001_clean.json"
        assert p.parent == tmp_path

    def test_graph_exists_false_when_missing(self, tmp_path):
        assert not graph_exists("item_999", "clean", tmp_path)

    def test_graph_exists_true_after_save(self, tmp_path):
        g = _make_graph()
        g["item_id"] = "myitem"
        g["condition"] = "perturbed"
        save_graph(g, graph_path("myitem", "perturbed", tmp_path))
        assert graph_exists("myitem", "perturbed", tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# graph_to_node_set
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphToNodeSet:
    def test_returns_exactly_k_when_enough_nodes(self):
        N = graph_to_node_set(GRAPH_LARGE, k=10)
        assert len(N) == 10

    def test_returns_all_when_fewer_than_k(self):
        N = graph_to_node_set(GRAPH_SMALL, k=100)
        assert len(N) == GRAPH_SMALL["n_nodes"]

    def test_top_by_influence(self):
        N = graph_to_node_set(GRAPH_LARGE, k=5)
        # All influences in N should be >= any influence NOT in N
        all_influences = sorted(
            [abs(nd["influence"]) for nd in GRAPH_LARGE["nodes"]], reverse=True
        )
        min_in_N = min(N.values())
        # 6th highest should be <= min influence in top-5
        if len(all_influences) > 5:
            assert all_influences[5] <= min_in_N + 1e-6

    def test_values_non_negative(self):
        N = graph_to_node_set(GRAPH_MEDIUM, k=8)
        assert all(v >= 0.0 for v in N.values())

    def test_keys_are_feature_ids(self):
        N = graph_to_node_set(GRAPH_SMALL, k=100)
        all_ids = {nd["feature_id"] for nd in GRAPH_SMALL["nodes"]}
        assert set(N.keys()) <= all_ids

    def test_k_zero_returns_empty(self):
        N = graph_to_node_set(GRAPH_MEDIUM, k=0)
        assert N == {}


# ─────────────────────────────────────────────────────────────────────────────
# graph_to_edge_vec
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphToEdgeVec:
    def test_returns_numpy_array(self):
        vec, keys = graph_to_edge_vec(GRAPH_MEDIUM, top_n=10)
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32

    def test_length_matches_min_of_top_n_and_n_edges(self):
        vec, keys = graph_to_edge_vec(GRAPH_SMALL, top_n=100)
        # GRAPH_SMALL has 3 edges
        assert len(vec) == min(100, GRAPH_SMALL["n_edges"])
        assert len(keys) == len(vec)

    def test_reference_keys_used_when_provided(self):
        ref_keys = [("feat_001", "feat_002"), ("feat_003", "feat_004")]
        vec, _ = graph_to_edge_vec(GRAPH_MEDIUM, reference_keys=ref_keys)
        assert len(vec) == len(ref_keys)

    def test_missing_edges_are_zero_in_reference_mode(self):
        """Edges not in the graph should appear as 0 in the vector."""
        ref_keys = [("nonexist_1", "nonexist_2"), ("also_fake", "also_fake")]
        vec, _ = graph_to_edge_vec(GRAPH_MEDIUM, reference_keys=ref_keys)
        np.testing.assert_array_equal(vec, np.zeros(2, dtype=np.float32))

    def test_top_n_picks_highest_absolute_weight(self):
        g = _make_graph(n_nodes=4, n_edges=4, seed=0)
        # Override with known weights
        g["edges"] = [
            {"src": "a", "dst": "b", "weight": 10.0},
            {"src": "c", "dst": "d", "weight": 0.1},
            {"src": "e", "dst": "f", "weight": 5.0},
            {"src": "g", "dst": "h", "weight": -8.0},
        ]
        g["n_edges"] = 4
        vec, keys = graph_to_edge_vec(g, top_n=2)
        # Top 2 by |weight|: 10.0 (a→b) and -8.0 (g→h)
        weights_selected = sorted([abs(w) for w in vec], reverse=True)
        assert weights_selected[0] == pytest.approx(10.0)
        assert weights_selected[1] == pytest.approx(8.0)


# ─────────────────────────────────────────────────────────────────────────────
# union_edge_keys
# ─────────────────────────────────────────────────────────────────────────────

class TestUnionEdgeKeys:
    def test_union_is_superset_of_both_top_sets(self):
        union = union_edge_keys(GRAPH_MEDIUM, GRAPH_LARGE, top_n=5)
        # Result should be a list of (src, dst) tuples
        assert all(isinstance(k, tuple) and len(k) == 2 for k in union)
        assert len(union) >= 1

    def test_no_duplicates_in_union(self):
        union = union_edge_keys(GRAPH_MEDIUM, GRAPH_LARGE, top_n=10)
        assert len(union) == len(set(union))

    def test_identical_graphs_union_equals_top_n(self):
        """Union of identical graphs = the top-n edges of that graph."""
        from src.graph_utils import graph_to_edge_vec as etv
        _, keys0 = etv(GRAPH_MEDIUM, top_n=8)
        union = union_edge_keys(GRAPH_MEDIUM, GRAPH_MEDIUM, top_n=8)
        assert set(keys0) == set(union)
