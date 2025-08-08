import math
from typing import Dict, List, Tuple

import torch
from torch import nn


class MultiRelationalGCNLayer(nn.Module):
    """
    Lightweight R-GCN layer without external graph libs.

    For each relation r, applies a linear transform and aggregates messages along edges.

    Inputs:
      - node_features: (num_nodes, in_dim)
      - edge_index_dict: dict[str, Tuple[Tensor, Tensor]] mapping relation -> (src_idx, dst_idx)
    Output:
      - (num_nodes, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, relation_names: List[str], add_self_loop: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relation_names = list(relation_names)
        self.add_self_loop = add_self_loop

        self.rel_linears = nn.ModuleDict({r: nn.Linear(in_dim, out_dim, bias=False) for r in self.relation_names})
        self.self_linear = nn.Linear(in_dim, out_dim, bias=True) if add_self_loop else None
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.ReLU()

    @staticmethod
    def aggregate_messages(h_src: torch.Tensor, dst_idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # h_src: (num_edges, out_dim) gathered features of source nodes after relation transform
        # dst_idx: (num_edges,) destination node indices
        out = torch.zeros(num_nodes, h_src.size(-1), device=h_src.device, dtype=h_src.dtype)
        out.index_add_(0, dst_idx, h_src)
        return out

    def forward(self, node_features: torch.Tensor, edge_index_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        num_nodes = node_features.size(0)
        messages = []
        for rel, (src_idx, dst_idx) in edge_index_dict.items():
            if src_idx.numel() == 0:
                continue
            h = self.rel_linears[rel](node_features)  # (N, out_dim)
            h_src = h.index_select(0, src_idx)        # (E, out_dim)
            agg = self.aggregate_messages(h_src, dst_idx, num_nodes)
            # simple degree normalization on destination side
            deg = torch.bincount(dst_idx, minlength=num_nodes).clamp(min=1).unsqueeze(-1)
            agg = agg / deg
            messages.append(agg)

        out = sum(messages) if messages else torch.zeros(num_nodes, self.out_dim, device=node_features.device)
        if self.add_self_loop and self.self_linear is not None:
            out = out + self.self_linear(node_features)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class MDGNN(nn.Module):
    """
    MDGNN = R-GCN over each snapshot + GRU over time per node.

    Given a window of length L of graphs and features, it produces predictions for next-step returns.

    Forward inputs:
      - x_seq_list: List[Tensor] length L of (N, in_dim) node features at times t-L+1..t
      - edges_seq_list: List[Dict[str, (src, dst)]] length L
    Output:
      - predictions: (N,) next-step return per node
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        relation_names: List[str],
        gnn_layers: int = 2,
        gru_hidden_dim: int = 64,
        mlp_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.relation_names = list(relation_names)
        self.gnn_layers = nn.ModuleList()
        last_dim = in_dim
        for _ in range(gnn_layers):
            self.gnn_layers.append(MultiRelationalGCNLayer(last_dim, hidden_dim, relation_names))
            last_dim = hidden_dim

        self.temporal_gru = nn.GRU(input_size=hidden_dim, hidden_size=gru_hidden_dim, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def encode_snapshot(self, x_t: torch.Tensor, edges_t: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        h = x_t
        for layer in self.gnn_layers:
            h = layer(h, edges_t)
        return h  # (N, hidden_dim)

    def forward(self, x_seq_list: List[torch.Tensor], edges_seq_list: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]):
        # Encode each snapshot with GNN
        encodings = []
        for x_t, edges_t in zip(x_seq_list, edges_seq_list):
            h_t = self.encode_snapshot(x_t, edges_t)  # (N, hidden_dim)
            encodings.append(h_t)
        # Stack over time -> (L, N, hidden_dim) -> (N, L, hidden_dim)
        H = torch.stack(encodings, dim=0).transpose(0, 1)
        # GRU over time with nodes as batch
        out, _ = self.temporal_gru(H)  # (N, L, gru_hidden_dim)
        last_hidden = out[:, -1, :]     # (N, gru_hidden_dim)
        pred = self.mlp(last_hidden).squeeze(-1)  # (N,)
        return pred