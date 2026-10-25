#!/usr/bin/env python3
"""Alternative graph encoders for Graph-M ablation (hypergraph vs bipartite vs pairwise)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class _ReactionFeatureMixin(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_dim: int,
        out_channels: int,
        dropout: float,
        reaction_feature_dim: int | None,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, conv_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(conv_dim, out_channels)
        self.reaction_proj = (
            nn.Linear(reaction_feature_dim, conv_dim) if reaction_feature_dim is not None else None
        )

    def _reaction_edge_attr(
        self,
        incidence_matrix: torch.Tensor,
        node_features: torch.Tensor,
        reaction_features: torch.Tensor | None,
    ) -> torch.Tensor:
        reaction_agg = torch.mm(incidence_matrix.T, node_features)
        if self.reaction_proj is None:
            return reaction_agg
        if reaction_features is None:
            zeros = torch.zeros(reaction_agg.size(0), self.reaction_proj.out_features, device=node_features.device)
            return torch.cat([reaction_agg, zeros], dim=-1)
        return torch.cat([reaction_agg, self.reaction_proj(reaction_features)], dim=-1)


class BipartiteCompoundEncoder(_ReactionFeatureMixin):
    """
    Reaction-metabolite bipartite message passing with GPR edge attributes.
    Metabolite nodes update from incident reactions; reaction attributes modulate messages.
    """

    def __init__(
        self,
        in_channels: int,
        conv_dim: int = 128,
        out_channels: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        reaction_feature_dim: int | None = None,
    ) -> None:
        super().__init__(in_channels, conv_dim, out_channels, dropout, reaction_feature_dim)
        self.num_layers = num_layers
        edge_in = conv_dim * 2 if reaction_feature_dim is not None else conv_dim
        self.edge_proj = nn.Linear(edge_in, conv_dim)
        self.conv_layers = nn.ModuleList(
            [GCNConv(conv_dim, conv_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        incidence_matrix: torch.Tensor,
        reaction_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = x.device
        x = F.relu(self.input_proj(x))
        num_compounds, num_reactions = incidence_matrix.shape
        compound_idx = torch.arange(num_compounds, device=device)
        reaction_idx = torch.arange(num_reactions, device=device) + num_compounds

        edge_attr = self.edge_proj(
            self._reaction_edge_attr(incidence_matrix, x, reaction_features)
        )
        src_list: list[torch.Tensor] = []
        dst_list: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for reaction in range(num_reactions):
            participants = torch.where(incidence_matrix[:, reaction] > 0)[0]
            if participants.numel() == 0:
                continue
            attr = edge_attr[reaction]
            src_list.append(reaction_idx[reaction].expand(participants.numel()))
            dst_list.append(participants)
            weights.append(attr.mean().expand(participants.numel()))

        if src_list:
            edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

        big_x = torch.cat([x, edge_attr], dim=0)
        for conv in self.conv_layers:
            big_x = F.relu(conv(big_x, edge_index))
            big_x = self.dropout(big_x)
        return self.output_proj(big_x[:num_compounds])


class PairwiseCompoundEncoder(_ReactionFeatureMixin):
    """Metabolite-only graph from co-participation in reactions (pairwise projection)."""

    def __init__(
        self,
        in_channels: int,
        conv_dim: int = 128,
        out_channels: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        reaction_feature_dim: int | None = None,
    ) -> None:
        super().__init__(in_channels, conv_dim, out_channels, dropout, reaction_feature_dim)
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList(
            [GCNConv(conv_dim, conv_dim) for _ in range(num_layers)]
        )

    @staticmethod
    def _pairwise_edge_index(incidence_matrix: torch.Tensor) -> torch.Tensor:
        device = incidence_matrix.device
        src: list[int] = []
        dst: list[int] = []
        num_compounds = incidence_matrix.size(0)
        for reaction in range(incidence_matrix.size(1)):
            nodes = torch.where(incidence_matrix[:, reaction] > 0)[0].tolist()
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    src.extend([nodes[i], nodes[j]])
                    dst.extend([nodes[j], nodes[i]])
        if not src:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        return torch.tensor([src, dst], dtype=torch.long, device=device)

    def forward(
        self,
        x: torch.Tensor,
        incidence_matrix: torch.Tensor,
        reaction_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        edge_index = self._pairwise_edge_index(incidence_matrix)
        for layer, conv in enumerate(self.conv_layers):
            x = conv(x if layer == 0 else x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        return self.output_proj(x)


def build_graph_encoder(
    encoder_type: str,
    *,
    in_channels: int,
    conv_dim: int,
    out_channels: int,
    num_layers: int,
    dropout: float,
    reaction_feature_dim: int | None,
) -> nn.Module:
    if encoder_type == "hypergraph":
        from .hypergraph_module import HypergraphCompoundEncoder

        return HypergraphCompoundEncoder(
            in_channels=in_channels,
            conv_dim=conv_dim,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            reaction_feature_dim=reaction_feature_dim,
        )
    if encoder_type == "bipartite":
        return BipartiteCompoundEncoder(
            in_channels=in_channels,
            conv_dim=conv_dim,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            reaction_feature_dim=reaction_feature_dim,
        )
    if encoder_type == "pairwise":
        return PairwiseCompoundEncoder(
            in_channels=in_channels,
            conv_dim=conv_dim,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            reaction_feature_dim=reaction_feature_dim,
        )
    raise ValueError(f"Unknown graph encoder type: {encoder_type}")
