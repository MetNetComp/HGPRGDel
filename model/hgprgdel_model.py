#!/usr/bin/env python3
"""
HGPRGDel Model: Gene Deletion Prediction with Hypergraph and GPR
=================================================================

Integrates:
- ESM-2 (gene) + ChemBERTa/MolBERT (metabolite) encoders
- Hypergraph neural network for reaction network
- GPR rule-based gene-compound association
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

try:
    from .gene_module import ESM2GeneEncoder
except ImportError:
    ESM2GeneEncoder = None
try:
    from .metabolite_module import get_metabolite_encoder
except ImportError:
    get_metabolite_encoder = None
from .hypergraph_module import HypergraphCompoundEncoder
from .gpr_module import GPRRuleParser, SoftBooleanAggregator, MeanPoolingGPRAggregator, GPRGeneCompoundAssociator, infer_num_reactions

try:
    from .graph_encoder_variants import build_graph_encoder
except ImportError:
    build_graph_encoder = None


class HGPRGDelHypergraphModel(nn.Module):
    """
    HGPRGDel model: LLM encoders + Hypergraph + GPR for gene deletion prediction.
    """

    def __init__(
        self,
        gene_dim: int = 320,
        metabolite_dim: int = 256,
        hypergraph_conv_dim: int = 128,
        hypergraph_num_layers: int = 2,
        gpr_assoc_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        use_chemberta: bool = True,
        freeze_gene_backbone: bool = True,
        freeze_metabolite_backbone: bool = True,
        use_hypergraph: bool = True,
        use_gpr: bool = True,
        gpr_aggregation: str = "soft_boolean",
        gpr_and_branch_mode: str = "combined",
        use_pretrained_encoders: bool = True,
        use_nma: bool = False,
        graph_encoder_type: str = "hypergraph",
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.metabolite_dim = metabolite_dim
        self.use_hypergraph = use_hypergraph
        self.use_gpr = use_gpr
        self.gpr_aggregation = gpr_aggregation
        self.gpr_and_branch_mode = gpr_and_branch_mode
        self.use_pretrained_encoders = use_pretrained_encoders
        self.use_nma = use_nma
        self.graph_encoder_type = graph_encoder_type if use_hypergraph else "none"

        if use_pretrained_encoders and ESM2GeneEncoder is not None:
            self.gene_encoder = ESM2GeneEncoder(
                hidden_size=gene_dim, dropout=dropout, freeze_backbone=freeze_gene_backbone
            )
        else:
            self.gene_encoder = _SimpleGeneEncoder(gene_dim)

        if use_pretrained_encoders and get_metabolite_encoder is not None:
            self.metabolite_encoder = get_metabolite_encoder(
                use_chemberta=use_chemberta,
                model_name="DeepChem/ChemBERTa-77M-MLM",
                hidden_size=metabolite_dim,
                dropout=dropout,
                freeze_backbone=freeze_metabolite_backbone,
            )
        else:
            self.metabolite_encoder = _SimpleMetaboliteEncoder(metabolite_dim)

        encoder_type = self.graph_encoder_type
        if use_hypergraph and build_graph_encoder is not None and encoder_type != "hypergraph":
            self.hypergraph = build_graph_encoder(
                encoder_type,
                in_channels=metabolite_dim,
                conv_dim=hypergraph_conv_dim,
                out_channels=metabolite_dim,
                num_layers=hypergraph_num_layers,
                dropout=dropout,
                reaction_feature_dim=gene_dim,
            )
        else:
            self.hypergraph = HypergraphCompoundEncoder(
                in_channels=metabolite_dim,
                conv_dim=hypergraph_conv_dim,
                out_channels=metabolite_dim,
                num_layers=hypergraph_num_layers,
                dropout=dropout,
                reaction_feature_dim=gene_dim,
            )

        self.gpr_parser = GPRRuleParser()
        if gpr_aggregation == "mean_pooling":
            self.gpr_aggregator = MeanPoolingGPRAggregator(gene_dim)
        else:
            self.gpr_aggregator = SoftBooleanAggregator(
                gene_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout,
                and_branch_mode=gpr_and_branch_mode,
            )
        self.gpr_associator = GPRGeneCompoundAssociator(
            gene_dim=gene_dim,
            compound_dim=metabolite_dim,
            out_dim=gpr_assoc_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(gene_dim + metabolite_dim + gpr_assoc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        gene_sequences: List[str],
        metabolite_smiles: List[str],
        reaction_compounds: Dict[int, List[int]],
        reaction_genes: Dict[int, List[int]],
        compound_reaction_edges: List[Tuple[int, int]],
        reaction_gene_edges: List[Tuple[int, int]],
        incidence_matrix: Optional[torch.Tensor] = None,
        gpr_rules: Optional[List[str]] = None,
        gene_to_idx: Optional[Dict[str, int]] = None,
        gpr_parsed: Optional[List[dict]] = None,
        gene_rxn_incidence: Optional[torch.Tensor] = None,
        compound_rxn_incidence: Optional[torch.Tensor] = None,
        batch_gene_idx: Optional[List[int]] = None,
        batch_metabolite_idx: Optional[List[int]] = None,
        return_reconstruction_loss: bool = False,
        gene_backbone_features: Optional[torch.Tensor] = None,
        metabolite_backbone_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        If batch_gene_idx and batch_metabolite_idx are given, use full gene_sequences and
        metabolite_smiles for encoding and index by batch. Else assume gene_sequences and
        metabolite_smiles are the batch (one-to-one).
        """
        device = next(self.parameters()).device

        gene_out = {}
        if gene_backbone_features is not None and hasattr(self.gene_encoder, "project_pooled_features"):
            gene_features = self.gene_encoder.project_pooled_features(gene_backbone_features)
        else:
            gene_out = self.gene_encoder(gene_sequences, return_features=True, return_reconstruction=return_reconstruction_loss)
            gene_features = gene_out["protein_features"] if isinstance(gene_out, dict) else gene_out

        meta_out = {}
        if metabolite_backbone_features is not None and hasattr(self.metabolite_encoder, "project_pooled_features"):
            metabolite_features = self.metabolite_encoder.project_pooled_features(metabolite_backbone_features)
        else:
            meta_out = self.metabolite_encoder(metabolite_smiles, return_features=True, return_reconstruction=return_reconstruction_loss)
            metabolite_features = meta_out["chemical_features"] if isinstance(meta_out, dict) else meta_out

        # GPR-M: compute reaction embeddings once (consumed by Graph-M and Pred-M)
        reaction_embeddings = None
        if self.use_gpr and gpr_rules and gene_to_idx and (reaction_compounds or reaction_genes):
            num_reactions = infer_num_reactions(reaction_compounds, reaction_genes, gpr_rules)
            reaction_emb = []
            for r_idx in range(num_reactions):
                if gpr_parsed is not None and r_idx < len(gpr_parsed):
                    parsed = gpr_parsed[r_idx]
                    if parsed.get("type") == "empty":
                        reaction_emb.append(
                            torch.zeros(self.gene_dim, device=device, dtype=gene_features.dtype)
                        )
                    else:
                        reaction_emb.append(self.gpr_aggregator(gene_features, parsed, gene_to_idx))
                else:
                    rule = gpr_rules[r_idx] if r_idx < len(gpr_rules) else ""
                    if rule and rule.strip():
                        parsed = self.gpr_parser.parse_gpr_rule(rule)
                        reaction_emb.append(self.gpr_aggregator(gene_features, parsed, gene_to_idx))
                    else:
                        reaction_emb.append(
                            torch.zeros(self.gene_dim, device=device, dtype=gene_features.dtype)
                        )
            reaction_embeddings = torch.stack(reaction_emb)

        if self.use_hypergraph and incidence_matrix is not None and incidence_matrix.numel() > 0:
            inc = incidence_matrix.to(device)
            compound_features = self.hypergraph(
                metabolite_features, inc, reaction_features=reaction_embeddings
            )
        elif self.use_nma and incidence_matrix is not None and incidence_matrix.numel() > 0:
            inc = incidence_matrix.to(device)
            compound_features = _neighborhood_mean_aggregation(metabolite_features, inc)
        else:
            compound_features = metabolite_features

        if batch_gene_idx is not None and batch_metabolite_idx is not None:
            g_idx = torch.tensor(batch_gene_idx, device=device, dtype=torch.long)
            c_idx = torch.tensor(batch_metabolite_idx, device=device, dtype=torch.long)
        else:
            batch_size = len(gene_sequences)
            g_idx = torch.arange(min(batch_size, gene_features.size(0)), device=device)
            c_idx = torch.arange(min(batch_size, compound_features.size(0)), device=device)
            if batch_size > g_idx.size(0):
                g_idx = torch.cat([g_idx, g_idx[-1:].expand(batch_size - g_idx.size(0))])
                c_idx = torch.cat([c_idx, c_idx[-1:].expand(batch_size - c_idx.size(0))])

        g_feat = gene_features[g_idx]
        c_feat = compound_features[c_idx]
        assoc_feat = self.gpr_associator.forward_batch_with_gpr(
            gene_features,
            compound_features,
            g_idx,
            c_idx,
            reaction_compounds=(reaction_compounds or {}) if self.use_gpr else {},
            reaction_genes=(reaction_genes or {}) if self.use_gpr else {},
            gpr_rules=gpr_rules if self.use_gpr else None,
            gpr_parser=self.gpr_parser,
            gpr_aggregator=self.gpr_aggregator,
            gene_to_idx=gene_to_idx if self.use_gpr else None,
            reaction_embeddings=reaction_embeddings,
            gpr_parsed=gpr_parsed if self.use_gpr else None,
            gene_rxn_incidence=gene_rxn_incidence if self.use_gpr else None,
            compound_rxn_incidence=compound_rxn_incidence if self.use_gpr else None,
        )
        combined = torch.cat([g_feat, c_feat, assoc_feat], dim=1)
        out = self.classifier(combined).squeeze(-1)

        if return_reconstruction_loss:
            recon_data = {}
            if isinstance(gene_out, dict) and "reconstruction_logits" in gene_out:
                recon_data["gene_reconstruction_logits"] = gene_out["reconstruction_logits"]
            if isinstance(meta_out, dict) and "reconstruction_logits" in meta_out:
                recon_data["metabolite_reconstruction_logits"] = meta_out["reconstruction_logits"]
            return out, recon_data
        return out


def _neighborhood_mean_aggregation(features: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
    """One-hop compound neighborhood mean over shared reaction hyperedges."""
    if features.size(0) != incidence_matrix.size(0):
        raise ValueError(
            "Neighborhood mean aggregation requires one feature row per full-GEM compound node: "
            f"got features={features.size(0)} and incidence_rows={incidence_matrix.size(0)}."
        )
    incidence = incidence_matrix.to(device=features.device, dtype=features.dtype)
    adjacency = incidence @ incidence.t()
    adjacency.fill_diagonal_(0)
    counts = adjacency.sum(dim=1, keepdim=True)
    aggregated = adjacency @ features
    has_neighbors = counts > 0
    averaged = aggregated / counts.clamp(min=1)
    return torch.where(has_neighbors, averaged, features)


class _SimpleGeneEncoder(nn.Module):
    """Ablation fallback: 1000-D position counts from first 20 AA symbols + Linear(1000, out_dim)."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(1000, out_dim)

    def forward(self, seqs, return_features=True, return_reconstruction=False):
        feats = []
        for s in seqs:
            v = torch.zeros(1000)
            for i, c in enumerate(s[:20]):
                v[i % 1000] += 1
            feats.append(self.proj(v.unsqueeze(0).to(next(self.proj.parameters()).device)))
        return {"protein_features": torch.cat(feats, dim=0)}


class _SimpleMetaboliteEncoder(nn.Module):
    """Ablation fallback: 1000-D SMILES char-ordinal features (first 20 chars) + Linear(1000, out_dim)."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(1000, out_dim)

    def forward(self, smiles_list, return_features=True, return_reconstruction=False):
        feats = []
        for s in smiles_list:
            v = torch.zeros(1000)
            for i, c in enumerate(str(s)[:20]):
                v[i % 1000] += ord(c) % 100
            feats.append(self.proj(v.unsqueeze(0).to(next(self.proj.parameters()).device)))
        out = {"chemical_features": torch.cat(feats, dim=0)}
        if return_reconstruction:
            out["reconstruction_logits"] = torch.zeros(len(smiles_list), 1000, device=out["chemical_features"].device)
        return out
