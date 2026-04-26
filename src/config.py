"""
src/config.py
Loads config.yaml and exposes a frozen Config dataclass.
All pipeline scripts call: cfg = load_config("config.yaml")
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Nested config dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Paths:
    data: str
    prompts: str
    pairs: str
    graphs: str
    activations: str
    agd: str
    behavioral: str
    analysis: str
    figures: str
    paper: str


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    dtype: Optional[str] = None
    device: str = "cuda"
    load_in_4bit: bool = False


@dataclasses.dataclass(frozen=True)
class Models:
    main: ModelSpec
    paraphrase: ModelSpec
    robustness: ModelSpec


@dataclasses.dataclass(frozen=True)
class Transcoders:
    hf_repo: str
    type: str
    layers: List[int]


@dataclasses.dataclass(frozen=True)
class AGDConfig:
    k: int
    alpha: float
    top_edges: int
    pruning_threshold: float


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    bbh_subtasks: List[str]
    bbh_items_per_subtask: int
    mmlu_categories: List[str]
    mmlu_items_per_cat: int
    gsm8k_items: int
    turpin_hint_items: int
    train_fraction: float


@dataclasses.dataclass(frozen=True)
class PilotConfig:
    n_items: int


@dataclasses.dataclass(frozen=True)
class BehavioralConfig:
    truncation_fractions: List[float]
    n_self_consistency: int


@dataclasses.dataclass(frozen=True)
class ParaphraseConfig:
    temperature: float
    min_edit_distance: int


@dataclasses.dataclass(frozen=True)
class GraphGenConfig:
    checkpoint_every: int
    max_new_tokens: int


@dataclasses.dataclass(frozen=True)
class StatsConfig:
    n_bootstrap: int
    bootstrap_method: str
    alpha_family: float
    h1_rho_threshold: float
    h2_auroc_threshold: float
    h3_delta_auroc: float


@dataclasses.dataclass(frozen=True)
class AblationsConfig:
    alpha_grid: List[float]
    k_grid: List[int]
    pruning_grid: List[float]
    layer_bands: Dict[str, List[int]]
    clt_subset_n: int
    robustness_n: int


@dataclasses.dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    models: Models
    transcoders: Transcoders
    agd: AGDConfig
    dataset: DatasetConfig
    pilot: PilotConfig
    behavioral: BehavioralConfig
    paraphrase: ParaphraseConfig
    graph_gen: GraphGenConfig
    stats: StatsConfig
    ablations: AblationsConfig
    _raw: Dict[str, Any] = dataclasses.field(compare=False, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "config.yaml") -> Config:
    """Load YAML config and return a frozen Config object.

    Parameters
    ----------
    path:
        Path to YAML config file. Defaults to ``config.yaml`` in the CWD.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # Ensure output directories exist
    p = raw["paths"]
    for key, dir_path in p.items():
        os.makedirs(dir_path, exist_ok=True)

    def _model(d: Dict) -> ModelSpec:
        return ModelSpec(
            name=d["name"],
            dtype=d.get("dtype"),
            device=d.get("device", "cuda"),
            load_in_4bit=d.get("load_in_4bit", False),
        )

    return Config(
        seed=raw["seed"],
        paths=Paths(**raw["paths"]),
        models=Models(
            main=_model(raw["models"]["main"]),
            paraphrase=_model(raw["models"]["paraphrase"]),
            robustness=_model(raw["models"]["robustness"]),
        ),
        transcoders=Transcoders(**raw["transcoders"]),
        agd=AGDConfig(**raw["agd"]),
        dataset=DatasetConfig(**raw["dataset"]),
        pilot=PilotConfig(**raw["pilot"]),
        behavioral=BehavioralConfig(**raw["behavioral"]),
        paraphrase=ParaphraseConfig(**raw["paraphrase"]),
        graph_gen=GraphGenConfig(**raw["graph_gen"]),
        stats=StatsConfig(**raw["stats"]),
        ablations=AblationsConfig(**raw["ablations"]),
        _raw=raw,
    )
