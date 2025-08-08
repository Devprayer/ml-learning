from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


RelationEdges = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class SyntheticStockDataset:
    features: torch.Tensor  # (T, N, F)
    targets: torch.Tensor   # (T, N) next-step returns; last step might be unused
    edges: List[RelationEdges]  # length T, edges at each t
    relation_names: List[str]

    @property
    def num_nodes(self) -> int:
        return self.features.size(1)

    @property
    def num_timesteps(self) -> int:
        return self.features.size(0)

    @property
    def feature_dim(self) -> int:
        return self.features.size(2)


def _build_static_edges_same_group(groups: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create undirected edges between nodes in the same group (both directions)
    src_list, dst_list = [], []
    n = len(groups)
    group_to_nodes: Dict[int, List[int]] = {}
    for i, g in enumerate(groups):
        group_to_nodes.setdefault(g, []).append(i)
    for nodes in group_to_nodes.values():
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                src_list.append(i)
                dst_list.append(j)
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(dst_list, dtype=torch.long)


def _build_topk_similarity_edges(emb: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Build edges based on cosine similarity top-k neighbors (directed both ways included by construction below)
    n = emb.shape[0]
    # Normalize
    norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_n = emb / norm
    sim = emb_n @ emb_n.T
    np.fill_diagonal(sim, -np.inf)
    src_list, dst_list = [], []
    for i in range(n):
        topk = np.argpartition(-sim[i], kth=min(k, n - 1) - 1)[:k]
        for j in topk:
            src_list.append(i)
            dst_list.append(int(j))
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(dst_list, dtype=torch.long)


def generate_synthetic_stock_data(
    num_stocks: int = 100,
    timesteps: int = 200,
    feature_dim: int = 16,
    num_relations: int = 3,
    sectors: int = 6,
    window_for_corr: int = 20,
    seed: int = 42,
) -> SyntheticStockDataset:
    """
    Generate a multi-relational dynamic graph dataset for stocks.

    Relations (example):
      - sector_same: static, connects stocks in the same sector
      - corr_topk: dynamic, connects top-k correlated returns in last window
      - style_similarity: static, connects similar style exposures
    """
    assert num_relations >= 2, "Use at least 2 relations"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    N, T, F = num_stocks, timesteps, feature_dim

    # Assign sectors
    sector_ids = [int(i % sectors) for i in range(N)]

    # Latent factors
    market_factor = rng.normal(0, 0.8, size=T)
    sector_factors = rng.normal(0, 1.0, size=(sectors, T))

    # Stock-specific AR(1) idiosyncratic component
    phi = 0.5
    epsilon = rng.normal(0, 0.5, size=(N, T))
    idio = np.zeros((N, T))
    for t in range(T):
        if t == 0:
            idio[:, t] = epsilon[:, t]
        else:
            idio[:, t] = phi * idio[:, t - 1] + epsilon[:, t]

    # Style exposures per stock (e.g., value, momentum)
    style_dim = 4
    style_expo = rng.normal(0, 1.0, size=(N, style_dim))
    style_factors = rng.normal(0, 0.5, size=(style_dim, T))

    # True returns generation
    true_returns = np.zeros((N, T))
    for t in range(T):
        for i in range(N):
            s = sector_ids[i]
            style_part = float(style_expo[i] @ style_factors[:, t])
            true_returns[i, t] = (
                0.4 * market_factor[t]
                + 0.6 * sector_factors[s, t]
                + 0.3 * style_part
                + idio[i, t]
            )

    # Node features: combine rolling stats of past returns and static descriptors
    static_desc = rng.normal(0, 1.0, size=(N, F // 2))
    features = np.zeros((T, N, F), dtype=np.float32)
    for t in range(T):
        past = true_returns[:, max(0, t - 10): t + 1]
        mean10 = past.mean(axis=1)
        std10 = past.std(axis=1) + 1e-6
        mom5 = past[:, -5:].mean(axis=1) if past.shape[1] >= 5 else mean10
        feat_dyn = np.stack([mean10, std10, mom5], axis=1)
        # Tile and noise to reach F // 2
        rep_dyn = np.tile(feat_dyn, (1, max(1, (F // 2 + 2) // 3)))[:, : F // 2]
        x_t = np.concatenate([rep_dyn, static_desc], axis=1).astype(np.float32)
        features[t] = x_t

    # Targets: next-step returns (shifted by -1 along time)
    targets = np.zeros((T, N), dtype=np.float32)
    targets[:-1, :] = true_returns[:, 1:].T
    targets[-1, :] = 0.0  # last step has no next-step info

    # Build relations
    relation_names = ["sector_same", "corr_topk"]
    if num_relations >= 3:
        relation_names.append("style_similarity")

    # Static edges
    sector_src, sector_dst = _build_static_edges_same_group(sector_ids)

    # Style similarity static edges via top-k in style_expo space
    style_src, style_dst = _build_topk_similarity_edges(style_expo, k=max(3, N // 20)) if "style_similarity" in relation_names else (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))

    # Dynamic correlation edges per timestep based on rolling window
    edges: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []
    for t in range(T):
        start = max(0, t - window_for_corr + 1)
        window = true_returns[:, start: t + 1]
        # Use simple correlation proxy: cosine similarity of demeaned returns over window
        demean = window - window.mean(axis=1, keepdims=True)
        src_corr, dst_corr = _build_topk_similarity_edges(demean, k=max(5, N // 15))
        rel_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
            "sector_same": (sector_src, sector_dst),
            "corr_topk": (src_corr, dst_corr),
        }
        if "style_similarity" in relation_names:
            rel_dict["style_similarity"] = (style_src, style_dst)
        edges.append(rel_dict)

    dataset = SyntheticStockDataset(
        features=torch.from_numpy(features),
        targets=torch.from_numpy(targets),
        edges=edges,
        relation_names=relation_names,
    )
    return dataset