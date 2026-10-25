#!/usr/bin/env python3
"""
GPR (Gene-Protein-Reaction) Rule Processing Module
=================================================

This module handles the parsing and processing of GPR rules in metabolic networks.
It converts Boolean logic GPR rules into differentiable neural network operations.

Key Components:
- GPRRuleParser: Parses Boolean GPR rules into structured format
- SoftBooleanAggregator: Converts Boolean logic to differentiable operations
- MeanPoolingGPRAggregator: Ablation baseline that mean-pools all rule genes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def infer_num_reactions(
    reaction_compounds: Optional[Dict[int, List[int]]],
    reaction_genes: Optional[Dict[int, List[int]]],
    gpr_rules: Optional[List[str]],
) -> int:
    """Infer reaction count (index range) from GPR / stoichiometry dicts and rule list."""
    n = len(gpr_rules) if gpr_rules else 0
    n = max(n, len(reaction_compounds or {}), len(reaction_genes or {}))
    for d in (reaction_compounds, reaction_genes):
        if d:
            n = max(n, max(d.keys()) + 1)
    return max(n, 1)


def parse_gpr_rules_list(gpr_rules: Optional[List[str]], parser: "GPRRuleParser") -> List[dict]:
    """Parse each GPR string once; forward passes only run the soft Boolean aggregator."""
    out: List[dict] = []
    for r in gpr_rules or []:
        if r and str(r).strip():
            out.append(parser.parse_gpr_rule(str(r)))
        else:
            out.append({"type": "empty", "genes": []})
    return out


def pad_gpr_parsed(parsed: List[dict], num_reactions: int) -> List[dict]:
    if num_reactions <= 0:
        return []
    if len(parsed) >= num_reactions:
        return parsed[:num_reactions]
    pad = [{"type": "empty", "genes": []} for _ in range(num_reactions - len(parsed))]
    return parsed + pad


def build_gene_compound_reaction_incidence(
    reaction_genes: Dict[int, List[int]],
    reaction_compounds: Dict[int, List[int]],
    num_genes: int,
    num_compounds: int,
    num_reactions: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dense participation masks: [num_genes, R] and [num_compounds, R], 1 if node participates in reaction.
    Used to vectorize R_g ∪ R_m aggregation in Pred-M.
    """
    g_mat = torch.zeros(num_genes, num_reactions, device=device, dtype=dtype)
    c_mat = torch.zeros(num_compounds, num_reactions, device=device, dtype=dtype)
    for r, genes in (reaction_genes or {}).items():
        if 0 <= int(r) < num_reactions:
            for g in genes:
                gi = int(g)
                if 0 <= gi < num_genes:
                    g_mat[gi, int(r)] = 1.0
    for r, comps in (reaction_compounds or {}).items():
        if 0 <= int(r) < num_reactions:
            for c in comps:
                ci = int(c)
                if 0 <= ci < num_compounds:
                    c_mat[ci, int(r)] = 1.0
    return g_mat, c_mat


def collect_gpr_gene_names(gpr_rule: Dict) -> List[str]:
    """Collect unique leaf gene names from a parsed GPR tree (order preserved)."""
    rule_type = gpr_rule.get("type")
    if rule_type == "empty":
        return []
    if rule_type == "gene":
        return list(gpr_rule.get("genes", []))

    names: List[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    if rule_type == "and":
        for item in gpr_rule.get("genes", []):
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                for gene_name in collect_gpr_gene_names(item):
                    _add(gene_name)
    elif rule_type == "or":
        for group in gpr_rule.get("groups", []):
            for gene_name in collect_gpr_gene_names(group):
                _add(gene_name)
        for item in gpr_rule.get("genes", []):
            if isinstance(item, str):
                _add(item)
            elif isinstance(item, dict):
                for gene_name in collect_gpr_gene_names(item):
                    _add(gene_name)
    return names

class GPRRuleParser:
    """Parser for Gene-Protein-Reaction (GPR) rules with Boolean logic."""
    
    def __init__(self):
        self.operators = {'and', 'or'}
        self.parentheses = {'(', ')'}
    
    def parse_gpr_rule(self, rule_string: str) -> Dict:
        """
        Parse GPR rule string into structured format.
        
        Examples:
        - "b0008 and b0009" -> {'type': 'and', 'genes': ['b0008', 'b0009']}
        - "b0014 or b0015" -> {'type': 'or', 'genes': ['b0014', 'b0015']}
        - "(b0020 and b0021) or b0022" -> {'type': 'or', 'groups': [{'type': 'and', 'genes': ['b0020', 'b0021']}, {'type': 'gene', 'genes': ['b0022']}]}
        
        Args:
            rule_string: GPR rule as string (e.g., "b0008 and b0009")
            
        Returns:
            Dictionary representing the parsed rule structure
        """
        if not rule_string or rule_string.strip() == '':
            return {'type': 'empty', 'genes': []}
        
        # Clean the rule string
        rule_string = rule_string.strip().lower()
        
        # Handle simple cases first
        if ' and ' in rule_string and ' or ' not in rule_string and '(' not in rule_string:
            genes = [g.strip() for g in rule_string.split(' and ')]
            return {'type': 'and', 'genes': genes}
        
        if ' or ' in rule_string and ' and ' not in rule_string and '(' not in rule_string:
            genes = [g.strip() for g in rule_string.split(' or ')]
            return {'type': 'or', 'genes': genes}
        
        # Handle complex nested rules
        return self._parse_complex_rule(rule_string)
    
    def _parse_complex_rule(self, rule_string: str) -> Dict:
        """Parse complex nested GPR rules with parentheses."""
        # Tokenize the rule
        tokens = self._tokenize(rule_string)
        
        # Parse using recursive descent
        result, _ = self._parse_expression(tokens, 0)
        return result
    
    def _tokenize(self, rule_string: str) -> List[str]:
        """Tokenize GPR rule string."""
        # Replace parentheses with spaces for easier parsing
        rule_string = rule_string.replace('(', ' ( ').replace(')', ' ) ')
        
        # Split by whitespace and filter empty strings
        tokens = [token.strip() for token in rule_string.split() if token.strip()]
        return tokens
    
    def _parse_expression(self, tokens: List[str], pos: int) -> Tuple[Dict, int]:
        """Parse expression using recursive descent."""
        if pos >= len(tokens):
            return {'type': 'empty', 'genes': []}, pos
        
        # Parse OR expression (lowest precedence)
        left, pos = self._parse_and_expression(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'or':
            pos += 1  # consume 'or'
            right, pos = self._parse_and_expression(tokens, pos)
            
            if left['type'] == 'or':
                left['groups'].append(right)
            else:
                left = {'type': 'or', 'groups': [left, right]}
        
        return left, pos
    
    def _parse_and_expression(self, tokens: List[str], pos: int) -> Tuple[Dict, int]:
        """Parse AND expression."""
        left, pos = self._parse_primary(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'and':
            pos += 1  # consume 'and'
            right, pos = self._parse_primary(tokens, pos)
            
            if left['type'] == 'and':
                if right['type'] == 'gene':
                    left['genes'].extend(right['genes'])
                else:
                    left['genes'].append(right)
            else:
                left = {'type': 'and', 'genes': [left]}
                if right['type'] == 'gene':
                    left['genes'].extend(right['genes'])
                else:
                    left['genes'].append(right)
        
        return left, pos
    
    def _parse_primary(self, tokens: List[str], pos: int) -> Tuple[Dict, int]:
        """Parse primary expression (gene or parenthesized expression)."""
        if pos >= len(tokens):
            return {'type': 'empty', 'genes': []}, pos
        
        if tokens[pos] == '(':
            pos += 1  # consume '('
            result, pos = self._parse_expression(tokens, pos)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1  # consume ')'
            return result, pos
        else:
            # Single gene
            gene = tokens[pos]
            pos += 1
            return {'type': 'gene', 'genes': [gene]}, pos


class SoftBooleanAggregator(nn.Module):
    """Soft Boolean aggregation for GPR rules using differentiable operations."""
    
    def __init__(
        self,
        gene_dim: int = 320,
        hidden_dim: int = 256,
        dropout_rate: float = 0.0,
        and_branch_mode: str = "combined",
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        if and_branch_mode not in ("combined", "mean_only", "soft_min_only"):
            raise ValueError(
                f"and_branch_mode must be combined, mean_only, or soft_min_only; got {and_branch_mode!r}"
            )
        self.and_branch_mode = and_branch_mode
        
        # Attention mechanism for OR aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=gene_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True,
        )

        # φ_∧: combines mean + soft-min for AND (enzyme-complex bottleneck semantics)
        self.combiner_and = nn.Sequential(
            nn.Linear(gene_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, gene_dim),
        )

        # φ_∨: combines attention + log-sum-exp for OR (isozyme sufficiency semantics)
        self.combiner_or = nn.Sequential(
            nn.Linear(gene_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, gene_dim),
        )
        
        # Learnable raw temperature; effective tau = softplus(raw) + eps (> 0)
        init_raw = float(torch.log(torch.exp(torch.tensor(1.0)) - 1.0))
        self.temperature = nn.Parameter(torch.tensor([init_raw]))
    
    def forward(self, gene_features: torch.Tensor, gpr_rule: Dict, gene_to_idx: Dict) -> torch.Tensor:
        """
        Apply soft Boolean aggregation to gene features based on GPR rule.
        
        Args:
            gene_features: [num_genes, gene_dim] tensor of gene embeddings
            gpr_rule: Parsed GPR rule dictionary
            gene_to_idx: Mapping from gene names to indices
            
        Returns:
            [gene_dim] tensor representing the aggregated reaction embedding
        """
        if gpr_rule['type'] == 'empty':
            return torch.zeros(self.gene_dim, device=gene_features.device)
        
        if gpr_rule['type'] == 'gene':
            # Single gene
            gene_name = gpr_rule['genes'][0]
            if gene_name in gene_to_idx:
                gene_idx = gene_to_idx[gene_name]
                # Check if gene index is within bounds of current batch
                if gene_idx < gene_features.size(0):
                    return gene_features[gene_idx]
                else:
                    return torch.zeros(self.gene_dim, device=gene_features.device)
            else:
                return torch.zeros(self.gene_dim, device=gene_features.device)
        
        if gpr_rule['type'] == 'and':
            return self._soft_and_aggregation(gene_features, gpr_rule['genes'], gene_to_idx)
        
        if gpr_rule['type'] == 'or':
            # Handle both simple OR (genes) and complex OR (groups)
            if 'groups' in gpr_rule:
                return self._soft_or_aggregation(gene_features, gpr_rule['groups'], gene_to_idx)
            elif 'genes' in gpr_rule:
                groups = [
                    {'type': 'gene', 'genes': [gene_name]}
                    for gene_name in gpr_rule['genes']
                ]
                return self._soft_or_aggregation(gene_features, groups, gene_to_idx)
        
        return torch.zeros(self.gene_dim, device=gene_features.device)
    
    def _soft_and_aggregation(self, gene_features: torch.Tensor, children: List, gene_to_idx: Dict) -> torch.Tensor:
        """Soft AND aggregation (φ_∧): mean + soft-min over recursively evaluated child subtrees.

        Each element of `children` is either a plain gene-name string (flat AND rule) or
        a parsed subtree dict (nested rule). All children are evaluated via self.forward()
        so that nested OR/AND subtrees are handled correctly, matching Algorithm 1.
        """
        child_vectors: List[torch.Tensor] = []
        for child in children:
            if isinstance(child, str):
                # Leaf gene name (flat AND rule, e.g. 'b0008')
                vec = self.forward(gene_features, {"type": "gene", "genes": [child]}, gene_to_idx)
            elif isinstance(child, dict):
                # Any subtree node (gene, and, or, empty) — fully recursive
                vec = self.forward(gene_features, child, gene_to_idx)
            else:
                continue
            child_vectors.append(vec)

        if not child_vectors:
            return torch.zeros(self.gene_dim, device=gene_features.device)

        # Stack child embeddings: [k, gene_dim]  (k=1 also goes through φ_∧, per Algorithm 1)
        gene_tensor = torch.stack(child_vectors)

        # Mean pooling: overall complex capability
        mean_pooled = torch.mean(gene_tensor, dim=0)

        # Coordinate-wise soft-min: bottleneck (weakest obligate subunit)
        tau = F.softplus(self.temperature) + 1e-6
        soft_min = -torch.logsumexp(-gene_tensor / tau, dim=0)

        # Ablation modes (single branch)
        if self.and_branch_mode == "mean_only":
            return mean_pooled
        if self.and_branch_mode == "soft_min_only":
            return soft_min

        # Combined: φ_∧([m ‖ s])
        combined = torch.cat([mean_pooled, soft_min], dim=0)
        return self.combiner_and(combined)
    
    def _soft_or_aggregation(self, gene_features: torch.Tensor, groups: List[Dict], gene_to_idx: Dict) -> torch.Tensor:
        """Soft OR aggregation (φ_∨): MHA + log-sum-exp over recursively evaluated child subtrees."""
        if not groups:
            return torch.zeros(self.gene_dim, device=gene_features.device)

        # Get embeddings for each group (k=1 also goes through φ_∨, per Algorithm 1)
        group_embeddings = []
        for group in groups:
            group_emb = self.forward(gene_features, group, gene_to_idx)
            group_embeddings.append(group_emb)
        
        # Stack group embeddings
        group_tensor = torch.stack(group_embeddings).unsqueeze(0)  # [1, num_groups, gene_dim]
        
        # Use attention to weight different groups (OR operation)
        # Query: average of all group embeddings
        query = torch.mean(group_tensor, dim=1, keepdim=True)  # [1, 1, gene_dim]
        
        # Key and Value: group embeddings
        attended, _ = self.attention(query, group_tensor, group_tensor)
        attended = attended.squeeze(0).squeeze(0)  # [gene_dim]
        
        # Also compute log-sum-exp (alternative OR approximation)
        logsumexp_or = torch.logsumexp(group_tensor.squeeze(0), dim=0)
        
        # Combined: φ_∨([a ‖ l])
        combined = torch.cat([attended, logsumexp_or], dim=0)
        return self.combiner_or(combined)


class MeanPoolingGPRAggregator(nn.Module):
    """Ablation: mean-pool all gene embeddings appearing in a GPR rule."""

    def __init__(self, gene_dim: int = 320):
        super().__init__()
        self.gene_dim = gene_dim

    def forward(
        self,
        gene_features: torch.Tensor,
        gpr_rule: Dict,
        gene_to_idx: Dict,
    ) -> torch.Tensor:
        if gpr_rule.get("type") == "empty":
            return torch.zeros(self.gene_dim, device=gene_features.device)

        embeddings = []
        for gene_name in collect_gpr_gene_names(gpr_rule):
            if gene_name not in gene_to_idx:
                continue
            gene_idx = gene_to_idx[gene_name]
            if gene_idx < gene_features.size(0):
                embeddings.append(gene_features[gene_idx])

        if not embeddings:
            return torch.zeros(self.gene_dim, device=gene_features.device)
        if len(embeddings) == 1:
            return embeddings[0]
        return torch.mean(torch.stack(embeddings), dim=0)


class GPRGeneCompoundAssociator(nn.Module):
    """
    Computes reaction-context features for gene-compound pairs from GPR rules.
    For each (gene, compound) pair, aggregates reaction embeddings over R_g ∪ R_m
    and projects via MLP. Uses only reaction features (no gene/compound concat)
    to avoid redundancy with the final prediction input.
    """

    def __init__(
        self,
        gene_dim: int,
        compound_dim: int,
        out_dim: int = 128,
        dropout: float = 0.0,
        reaction_context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.compound_dim = compound_dim
        self.out_dim = out_dim
        self.reaction_context_dim = reaction_context_dim if reaction_context_dim is not None else gene_dim
        # MLP uses only reaction context (c_rxn), not gene/compound features
        self.assoc_mlp = nn.Sequential(
            nn.Linear(self.reaction_context_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )

    def forward(
        self,
        gene_features: torch.Tensor,
        compound_features: torch.Tensor,
        reaction_compounds: Dict[int, List[int]],
        reaction_genes: Dict[int, List[int]],
        gpr_rules: List[str],
        gpr_parser,
        gpr_aggregator,
        gene_to_idx: Dict,
        gene_idx: int,
        compound_idx: int,
    ) -> torch.Tensor:
        """
        Compute association feature for a single (gene, compound) pair.
        Uses reaction context (R_g ∪ R_m) only; falls back to zeros if no GPR.
        """
        device = gene_features.device
        num_reactions = max(len(reaction_compounds), len(reaction_genes), len(gpr_rules) if gpr_rules else 0)
        if num_reactions == 0 or not gpr_rules or not gene_to_idx:
            ctx = torch.zeros(self.reaction_context_dim, device=device, dtype=gene_features.dtype)
        else:
            reaction_emb = []
            for r_idx in range(num_reactions):
                if r_idx >= len(gpr_rules) or not gpr_rules[r_idx]:
                    emb = torch.zeros(self.reaction_context_dim, device=device, dtype=gene_features.dtype)
                else:
                    parsed = gpr_parser.parse_gpr_rule(gpr_rules[r_idx])
                    emb = gpr_aggregator(gene_features, parsed, gene_to_idx)
                reaction_emb.append(emb)
            reaction_emb_t = torch.stack(reaction_emb)
            R_g = {r for r in range(num_reactions) if gene_idx in reaction_genes.get(r, [])}
            R_m = {r for r in range(num_reactions) if compound_idx in reaction_compounds.get(r, [])}
            R_union = R_g | R_m
            if R_union:
                ctx = reaction_emb_t[list(R_union)].mean(dim=0)
            else:
                ctx = torch.zeros(self.reaction_context_dim, device=device, dtype=gene_features.dtype)
        return self.assoc_mlp(ctx.unsqueeze(0)).squeeze(0)

    def forward_batch(
        self,
        gene_features: torch.Tensor,
        compound_features: torch.Tensor,
        gene_indices: torch.Tensor,
        compound_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Batch version without GPR: uses zeros for reaction_context."""
        device = gene_features.device
        batch_size = gene_indices.size(0)
        zeros_ctx = torch.zeros(batch_size, self.reaction_context_dim, device=device, dtype=gene_features.dtype)
        return self._assoc_forward(
            gene_features, compound_features, gene_indices, compound_indices, zeros_ctx
        )

    def forward_batch_with_gpr(
        self,
        gene_features: torch.Tensor,
        compound_features: torch.Tensor,
        gene_indices: torch.Tensor,
        compound_indices: torch.Tensor,
        reaction_compounds: Dict[int, List[int]],
        reaction_genes: Dict[int, List[int]],
        gpr_rules: Optional[List[str]],
        gpr_parser,
        gpr_aggregator,
        gene_to_idx: Optional[Dict[str, int]],
        reaction_embeddings: Optional[torch.Tensor] = None,
        gpr_parsed: Optional[List[dict]] = None,
        gene_rxn_incidence: Optional[torch.Tensor] = None,
        compound_rxn_incidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch version with GPR: aggregates reaction embeddings over R_g ∪ R_m
        and projects to reaction-context features (Pred-M input preparation).
        If reaction_embeddings is provided (from GPR-M), uses it; otherwise
        computes via gpr_parser and gpr_aggregator. If GPR data is missing,
        uses zeros for reaction_context.
        """
        device = gene_features.device
        batch_size = gene_indices.size(0)

        use_gpr = (
            (reaction_embeddings is not None and reaction_embeddings.numel() > 0)
            or (
                gpr_rules is not None
                and len(gpr_rules) > 0
                and gene_to_idx is not None
                and len(gene_to_idx) > 0
                and reaction_compounds is not None
                and reaction_genes is not None
                and (len(reaction_compounds) > 0 or len(reaction_genes) > 0)
            )
        )

        if not use_gpr:
            zeros_ctx = torch.zeros(batch_size, self.reaction_context_dim, device=device, dtype=gene_features.dtype)
            return self._assoc_forward(
                gene_features, compound_features, gene_indices, compound_indices, zeros_ctx
            )

        num_reactions = max(
            len(reaction_compounds),
            len(reaction_genes),
            len(gpr_rules) if gpr_rules else 0,
            reaction_embeddings.size(0) if reaction_embeddings is not None else 0,
        )

        if reaction_embeddings is not None:
            re = reaction_embeddings.to(device, dtype=gene_features.dtype)
            if re.size(0) >= num_reactions:
                reaction_emb_t = re[:num_reactions]
            else:
                pad = torch.zeros(num_reactions - re.size(0), re.size(1), device=device, dtype=re.dtype)
                reaction_emb_t = torch.cat([re, pad], dim=0)
        else:
            reaction_emb = []
            rules = gpr_rules or []
            for r_idx in range(num_reactions):
                if gpr_parsed is not None and r_idx < len(gpr_parsed):
                    parsed = gpr_parsed[r_idx]
                    if parsed.get("type") == "empty":
                        emb = torch.zeros(self.reaction_context_dim, device=device, dtype=gene_features.dtype)
                    else:
                        emb = gpr_aggregator(gene_features, parsed, gene_to_idx)
                elif r_idx >= len(rules) or not rules[r_idx]:
                    emb = torch.zeros(self.reaction_context_dim, device=device, dtype=gene_features.dtype)
                else:
                    parsed = gpr_parser.parse_gpr_rule(rules[r_idx])
                    emb = gpr_aggregator(gene_features, parsed, gene_to_idx)
                reaction_emb.append(emb)
            reaction_emb_t = torch.stack(reaction_emb)

        if (
            gene_rxn_incidence is not None
            and compound_rxn_incidence is not None
            and gene_rxn_incidence.size(1) >= num_reactions
            and compound_rxn_incidence.size(1) >= num_reactions
        ):
            g_i = gene_indices.to(device=device, dtype=torch.long)
            c_i = compound_indices.to(device=device, dtype=torch.long)
            m_g = gene_rxn_incidence[g_i, :num_reactions]
            m_c = compound_rxn_incidence[c_i, :num_reactions]
            m_union = (m_g + m_c) > 0
            rdt = reaction_emb_t.dtype
            weighted = reaction_emb_t.unsqueeze(0) * m_union.unsqueeze(-1).to(rdt)
            sums = weighted.sum(dim=1)
            denom = m_union.sum(dim=1, keepdim=True).clamp(min=1).to(rdt)
            reaction_context = sums / denom
        else:
            reaction_context_list = []
            g_idx_np = gene_indices.cpu().numpy()
            c_idx_np = compound_indices.cpu().numpy()
            for i in range(batch_size):
                g_idx = int(g_idx_np[i])
                c_idx = int(c_idx_np[i])
                R_g = {r for r in range(num_reactions) if g_idx in reaction_genes.get(r, [])}
                R_m = {r for r in range(num_reactions) if c_idx in reaction_compounds.get(r, [])}
                R_union = R_g | R_m
                if R_union:
                    ctx = reaction_emb_t[list(R_union)].mean(dim=0)
                else:
                    ctx = torch.zeros(self.reaction_context_dim, device=device, dtype=gene_features.dtype)
                reaction_context_list.append(ctx)
            reaction_context = torch.stack(reaction_context_list)

        return self._assoc_forward(
            gene_features, compound_features, gene_indices, compound_indices, reaction_context
        )

    def _assoc_forward(
        self,
        gene_features: torch.Tensor,
        compound_features: torch.Tensor,
        gene_indices: torch.Tensor,
        compound_indices: torch.Tensor,
        reaction_context: torch.Tensor,
    ) -> torch.Tensor:
        """Internal: project reaction_context only (no gene/compound concat)."""
        return self.assoc_mlp(reaction_context)

