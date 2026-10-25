#!/usr/bin/env python3
"""
Hypergraph Neural Network Module for HGPRGDel
=============================================

Uses hypergraph convolution to represent metabolic reaction networks.
Each reaction is a hyperedge connecting its substrate and product compounds.
Incorporates GPR-derived reaction features as hyperedge attributes for gene-metabolite fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from typing import Optional, Tuple


class HypergraphCompoundEncoder(nn.Module):
    """
    Hypergraph neural network for compound (metabolite) representation.
    Nodes = compounds, Hyperedges = reactions.
    Uses reaction features (GPR embeddings) as hyperedge attributes with attention
    to fuse gene information into metabolite representations.
    """

    def __init__(
        self,
        in_channels: int,
        conv_dim: int = 128,
        out_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        reaction_feature_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_dim = conv_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.reaction_feature_dim = reaction_feature_dim

        self.input_proj = nn.Linear(in_channels, conv_dim)

        hyperedge_in_dim = conv_dim
        if reaction_feature_dim is not None:
            self.reaction_proj = nn.Linear(reaction_feature_dim, conv_dim)
            hyperedge_in_dim = conv_dim * 2
        else:
            self.reaction_proj = None

        self.hyperedge_proj = nn.Linear(hyperedge_in_dim, conv_dim)

        self.hypergraph_conv = HypergraphConv(
            in_channels=conv_dim,
            out_channels=conv_dim,
            use_attention=True,
            attention_mode="node",
            heads=1,
            concat=False,
        )

        self.hypergraph_conv_list = nn.ModuleList()
        self.hyperedge_proj_list = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hypergraph_conv_list.append(
                HypergraphConv(
                    in_channels=conv_dim,
                    out_channels=conv_dim,
                    use_attention=True,
                    attention_mode="node",
                    heads=1,
                    concat=False,
                )
            )
            self.hyperedge_proj_list.append(nn.Linear(hyperedge_in_dim, conv_dim))

        self.output_proj = nn.Linear(conv_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        incidence_matrix: torch.Tensor,
        reaction_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Compound features [num_compounds, in_channels]
            incidence_matrix: [num_compounds, num_reactions], 1 if compound participates in reaction
            reaction_features: Optional [num_reactions, reaction_feature_dim] GPR-derived reaction embeddings

        Returns:
            Updated compound features [num_compounds, out_channels]
        """
        device = x.device
        x = F.relu(self.input_proj(x))

        row, col = torch.where(incidence_matrix.T)
        row = row.to(device)
        col = col.to(device)
        edges = torch.stack([col, row], dim=0)

        reaction_agg = torch.mm(incidence_matrix.T, x)
        if self.reaction_proj is not None:
            if reaction_features is not None:
                gpr_proj = self.reaction_proj(reaction_features)
                hyperedge_input = torch.cat([reaction_agg, gpr_proj], dim=-1)
            else:
                zeros_gpr = torch.zeros(reaction_agg.size(0), self.conv_dim, device=x.device, dtype=x.dtype)
                hyperedge_input = torch.cat([reaction_agg, zeros_gpr], dim=-1)
        else:
            hyperedge_input = reaction_agg
        hyperedge_attr = self.hyperedge_proj(hyperedge_input)

        x = self.hypergraph_conv(x, edges, hyperedge_attr=hyperedge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        for i in range(self.num_layers - 1):
            reaction_agg = torch.mm(incidence_matrix.T, x)
            if self.reaction_proj is not None:
                if reaction_features is not None:
                    gpr_proj = self.reaction_proj(reaction_features)
                    hyperedge_input = torch.cat([reaction_agg, gpr_proj], dim=-1)
                else:
                    zeros_gpr = torch.zeros(reaction_agg.size(0), self.conv_dim, device=x.device, dtype=x.dtype)
                    hyperedge_input = torch.cat([reaction_agg, zeros_gpr], dim=-1)
            else:
                hyperedge_input = reaction_agg
            hyperedge_attr = self.hyperedge_proj_list[i](hyperedge_input)
            x = self.hypergraph_conv_list[i](x, edges, hyperedge_attr=hyperedge_attr)
            x = F.relu(x)
            x = self.dropout(x)

        return self.output_proj(x)
