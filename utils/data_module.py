#!/usr/bin/env python3
"""
Data Loading and Preprocessing Module for HGPRGDel
===================================================

Loads COBRA-style JSON, per-gene sequences, target-metabolite SMILES, and labels for three GEMs:
e_coli_core, iML1515, iMM904.

Key Components:
- MetabolicDataset, BalancedMetabolicDataset
- load_metabolic_data_for_gem: load from ``Data/<GEM>/`` (JSON, Gene_AA, Label, SMILES table)
- write_metabolite_gene_predictions_csv: write long-form prediction CSV
"""

import os
import json
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Iterable, List, Tuple, Optional

from utils.prediction_threshold import (
    DEFAULT_BINARY_THRESHOLD_MODE,
    BinaryThresholdMode,
    binary_predictions_for_pairs,
)

# High-flux / currency metabolites often excluded from reaction-hyperedge construction (KEGG/BiGG-style IDs)
CURRENCY_METABOLITES = {
    "atp", "adp", "amp", "nad", "nadh", "nadp", "nadph", "coa", "h", "h2o",
    "pi", "ppi", "co2", "nh4", "o2", "glutathione", "na1", "k", "mg2", "ca2"
}


def _is_currency(metabolite_id: str) -> bool:
    """Check if metabolite is in currency list (case-insensitive, strip compartment)."""
    base = metabolite_id.split("_")[0].lower()
    return base in CURRENCY_METABOLITES or metabolite_id.lower() in CURRENCY_METABOLITES


def _metabolite_id_keys(metabolite_id: str) -> set:
    """Return conservative matching keys for COBRA/BiGG metabolite IDs."""
    met = str(metabolite_id).strip().lower()
    keys = {met}
    if "_" in met:
        base, compartment = met.rsplit("_", 1)
        if 1 <= len(compartment) <= 2:
            keys.add(base)
    return keys


class MetabolicDataset(Dataset):
    """Dataset for metabolic data with gene-metabolite pairs."""

    def __init__(self, gene_sequences, metabolite_smiles, task_labels, gene_names, metabolite_names,
                 reaction_compounds, reaction_genes, compound_reaction_edges, reaction_gene_edges, gpr_rules, gene_to_idx,
                 compound_names=None, target_to_compound_idx=None):
        self.gene_sequences = gene_sequences
        self.metabolite_smiles = metabolite_smiles
        self.task_labels = task_labels
        self.gene_names = gene_names
        # metabolite_names are the labeled target metabolites used for prediction/CV.
        self.metabolite_names = metabolite_names
        # compound_names are the full-GEM metabolite nodes consumed by Graph-M.
        self.compound_names = compound_names if compound_names is not None else metabolite_names
        self.target_to_compound_idx = target_to_compound_idx or {
            m: self.compound_names.index(m)
            for m in metabolite_names
            if m in self.compound_names
        }
        self.reaction_compounds = reaction_compounds
        self.reaction_genes = reaction_genes
        self.compound_reaction_edges = compound_reaction_edges
        self.reaction_gene_edges = reaction_gene_edges
        self.gpr_rules = gpr_rules
        self.gene_to_idx = gene_to_idx
        self.pairs = [(g, m) for g in gene_names for m in metabolite_names]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gene_name, metabolite_name = self.pairs[idx]
        gene_idx = self.gene_names.index(gene_name) if gene_name in self.gene_names else 0
        metabolite_idx = self.target_to_compound_idx.get(metabolite_name, 0)
        return {
            'gene_sequence': self.gene_sequences[gene_idx],
            'metabolite_smiles': self.metabolite_smiles[metabolite_idx],
            'label': self.task_labels.get((gene_name, metabolite_name), 0),
            'gene_name': gene_name,
            'metabolite_name': metabolite_name,
            'compound_idx': metabolite_idx,
        }


class BalancedMetabolicDataset(MetabolicDataset):
    """Balanced dataset for metabolic data."""

    def __init__(self, gene_sequences, metabolite_smiles, task_labels, gene_names, metabolite_names,
                 reaction_compounds, reaction_genes, compound_reaction_edges, reaction_gene_edges,
                 gpr_rules, gene_to_idx, balance_strategy='oversample', random_seed=42,
                 compound_names=None, target_to_compound_idx=None):
        super().__init__(gene_sequences, metabolite_smiles, task_labels, gene_names, metabolite_names,
                         reaction_compounds, reaction_genes, compound_reaction_edges, reaction_gene_edges,
                         gpr_rules, gene_to_idx, compound_names=compound_names,
                         target_to_compound_idx=target_to_compound_idx)
        self.pairs = self._create_balanced_pairs(balance_strategy, random_seed)
        from collections import Counter
        labels = [self.task_labels.get(p, 0) for p in self.pairs]
        cnt = Counter(labels)
        print(f"Balanced dataset: DELETE={cnt[0]}, KEEP={cnt[1]}")

    def _create_balanced_pairs(self, balance_strategy, random_seed):
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        delete_pairs = [(g, m) for g in self.gene_names for m in self.metabolite_names
                        if self.task_labels.get((g, m), 0) == 0]
        keep_pairs = [(g, m) for g in self.gene_names for m in self.metabolite_names
                      if self.task_labels.get((g, m), 0) == 1]
        if balance_strategy == 'oversample':
            if len(delete_pairs) < len(keep_pairs):
                delete_pairs = random.choices(delete_pairs, k=len(keep_pairs))
            else:
                keep_pairs = random.choices(keep_pairs, k=len(delete_pairs))
        elif balance_strategy == 'undersample':
            n = min(len(delete_pairs), len(keep_pairs))
            delete_pairs = random.sample(delete_pairs, n)
            keep_pairs = random.sample(keep_pairs, n)
        pairs = delete_pairs + keep_pairs
        random.shuffle(pairs)
        return pairs


PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD = 0.9


def _gene_label_counts_from_task_labels(task_labels: Dict) -> Dict[str, Dict[int, int]]:
    counts: Dict[str, Dict[int, int]] = {}
    for (gene, _metabolite), label in task_labels.items():
        if gene not in counts:
            counts[gene] = {0: 0, 1: 0}
        counts[gene][label] = counts[gene].get(label, 0) + 1
    return counts


def get_predetermined_gene_set(
    counts: Dict[str, Dict[int, int]],
    threshold: float = PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD,
) -> set:
    """Genes whose supervised label is identical for >= threshold of target metabolites."""
    predetermined: set = set()
    for gene, tally in counts.items():
        total = sum(tally.values())
        if total and max(tally.values()) / total >= threshold:
            predetermined.add(gene)
    return predetermined


def get_evaluation_genes(
    gene_names: List[str],
    task_labels: Dict,
    metabolite_names: List[str],
    threshold: float = PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD,
) -> List[str]:
    """Return genes included in reported evaluation metrics.

    Pre-determined genes (label consistency >= threshold across supervised targets)
    remain in the training label matrix but are excluded from metric computation,
    matching the baseline evaluation protocol.
    """
    counts = _gene_label_counts_from_task_labels(task_labels)
    predetermined = get_predetermined_gene_set(counts, threshold=threshold)
    eval_genes = [g for g in gene_names if g in counts and g not in predetermined]
    print(
        f"Evaluation genes: {len(eval_genes)} "
        f"(excluded {len(predetermined)} pre-determined at >={threshold:.0%} consistency)"
    )
    return eval_genes


def get_predetermined_genes_from_prediction_rows(
    rows: List[dict],
    threshold: float = PREDETERMINED_LABEL_CONSISTENCY_THRESHOLD,
) -> set:
    """Identify pre-determined genes from long-form prediction CSV rows."""
    counts: Dict[str, Dict[int, int]] = {}
    for row in rows:
        try:
            label = int(float(row["True_Label"]))
        except (KeyError, TypeError, ValueError):
            continue
        if label not in (0, 1):
            continue
        gene = row["Gene"]
        if gene not in counts:
            counts[gene] = {0: 0, 1: 0}
        counts[gene][label] = counts[gene].get(label, 0) + 1
    return get_predetermined_gene_set(counts, threshold=threshold)


def _load_json_model(data_path: str, cbm: str) -> Optional[dict]:
    """Load COBRA JSON model. Tries {cbm}.json in data_path."""
    candidates = [
        os.path.join(data_path, f"{cbm}.json"),
        os.path.join(data_path, cbm, f"{cbm}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    return None


def _load_metabolite_smiles_csv(path: str) -> Tuple[List[str], List[str]]:
    """Load metabolite IDs and SMILES from ``Target metabolites_smile.csv``-style tables."""
    df = pd.read_csv(path)
    col_met = 'Metabolite' if 'Metabolite' in df.columns else df.columns[0]
    col_smi = 'Smiles' if 'Smiles' in df.columns else ('SMILES' if 'SMILES' in df.columns else df.columns[1])
    names = df[col_met].astype(str).tolist()
    smiles = df[col_smi].fillna('').astype(str).tolist()
    return names, smiles


def load_metabolic_data_for_gem(
    data_root: str,
    cbm: str,
    filter_currency: bool = True,
    restore_target_metabolites: bool = True,
) -> Tuple:
    """
    Load metabolic data for a given GEM from ``data_root`` / ``Data/<GEM>/``.

    Args:
        data_root: Base path containing Data/ or direct path to CBM folder
        cbm: One of 'e_coli_core', 'iML1515', 'iMM904'
        filter_currency: If True, exclude currency metabolites from reaction-compound mappings
        restore_target_metabolites: If True, always include target metabolites in compound set

    Returns:
        Same tuple as load_metabolic_data: (gene_sequences, metabolite_smiles, task_labels,
        gene_names, metabolite_names, reaction_compounds, reaction_genes, compound_reaction_edges,
        reaction_gene_edges, gpr_rules, gene_to_idx)
        Plus: incidence_matrix (torch.Tensor), num_compounds, num_reactions
    """
    data_path = os.path.join(data_root, cbm) if os.path.isdir(data_root) else data_root
    if not os.path.exists(data_path):
        data_path = os.path.join(data_root, "Data", cbm)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    print(f"Loading {cbm} from {data_path}")

    # 1. Gene sequences (Gene_AA/*.csv, column 'AA Sequence')
    gene_aa_path = os.path.join(data_path, "Gene_AA")
    gene_names, gene_sequences = [], []
    if os.path.exists(gene_aa_path):
        for f in os.listdir(gene_aa_path):
            if f.endswith('.csv') and not f.endswith('.zip'):
                p = os.path.join(gene_aa_path, f)
                try:
                    df = pd.read_csv(p)
                    if 'AA Sequence' in df.columns and len(df) > 0:
                        seq = str(df.iloc[0]['AA Sequence'])
                        if seq and seq != 'nan':
                            gene_names.append(f.replace('.csv', ''))
                            gene_sequences.append(seq)
                except Exception as e:
                    print(f"Warning: skip {f}: {e}")
    print(f"Genes: {len(gene_sequences)}")

    # 2. Target metabolite SMILES define labeled prediction tasks.
    target_smiles_path = os.path.join(data_path, "Target metabolites_smile.csv")
    metabolite_names, target_metabolite_smiles = [], []
    if os.path.exists(target_smiles_path):
        metabolite_names, target_metabolite_smiles = _load_metabolite_smiles_csv(target_smiles_path)
    print(f"Target metabolites: {len(metabolite_names)}")

    # 3. Labels (Label/*.csv, columns Gene, Deleted)
    label_path = os.path.join(data_path, "Label")
    task_labels = {}
    if os.path.exists(label_path):
        for f in os.listdir(label_path):
            if f.endswith('.csv') and not f.endswith('.zip'):
                p = os.path.join(label_path, f)
                df = pd.read_csv(p)
                meta = f.replace('.csv', '')
                gcol = 'Gene' if 'Gene' in df.columns else df.columns[0]
                lcol = 'Deleted' if 'Deleted' in df.columns else df.columns[1]
                for _, row in df.iterrows():
                    g = str(row[gcol])
                    lab = int(row[lcol]) if pd.notna(row[lcol]) else 0
                    task_labels[(g, meta)] = lab
    print(f"Labels: {len(task_labels)}")

    # 4. Metabolic model JSON for reactions, GPR
    base_root = os.path.dirname(data_path) if os.path.basename(data_path) == cbm else data_path
    model_data = _load_json_model(base_root, cbm)
    if model_data is None:
        model_data = _load_json_model(data_path, cbm)

    reactions = []
    gpr_rules = []
    if model_data:
        rxns = model_data.get('reactions', model_data.get('reaction', []))
        for r in rxns:
            mid = r.get('id', r.get('reaction_id', ''))
            mets = r.get('metabolites', r.get('metabolite_ids', {}))
            if isinstance(mets, list):
                mets = {m: 1 for m in mets}
            grr = r.get('gene_reaction_rule', r.get('gene_reaction_rule', '')) or ''
            reactions.append({'id': mid, 'metabolites': mets, 'gene_reaction_rule': grr})
            gpr_rules.append(grr)
        print(f"Reactions: {len(reactions)}")
    else:
        print("No JSON model found; reaction_compounds and GPR will be empty.")

    # 5. Full-GEM compound SMILES. Prediction remains target-only, but Graph-M uses
    # all GEM metabolites as compound nodes.
    full_smiles_path = os.path.join(data_path, "All metabolites_smile.csv")
    if os.path.exists(full_smiles_path):
        compound_names, metabolite_smiles = _load_metabolite_smiles_csv(full_smiles_path)
    elif model_data:
        compound_names = [
            m.get('id', '')
            for m in model_data.get('metabolites', [])
            if m.get('id')
        ]
        target_smiles_by_key = {}
        for target, smiles in zip(metabolite_names, target_metabolite_smiles):
            for key in _metabolite_id_keys(target):
                target_smiles_by_key.setdefault(key, smiles)
        metabolite_smiles = [
            next((target_smiles_by_key[key] for key in _metabolite_id_keys(met) if key in target_smiles_by_key), '')
            for met in compound_names
        ]
        print(
            "Warning: All metabolites_smile.csv not found; full-GEM graph uses JSON metabolite IDs "
            "with SMILES only for matched targets."
        )
    else:
        compound_names = list(metabolite_names)
        metabolite_smiles = list(target_metabolite_smiles)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    compound_to_idx = {m: i for i, m in enumerate(compound_names)}
    target_metabolites = set(metabolite_names)  # labels define target metabolites
    target_key_set = set()
    for target_name in metabolite_names:
        target_key_set.update(_metabolite_id_keys(target_name))
    compound_key_to_idx = {}
    for cname, ci in compound_to_idx.items():
        for key in _metabolite_id_keys(cname):
            compound_key_to_idx.setdefault(key, ci)
    target_to_compound_idx = {}
    for target_name in metabolite_names:
        matching_keys = _metabolite_id_keys(target_name) & set(compound_key_to_idx)
        if matching_keys:
            target_to_compound_idx[target_name] = compound_key_to_idx[sorted(matching_keys)[0]]

    def should_include_compound(met_id: str) -> bool:
        if restore_target_metabolites:
            if _metabolite_id_keys(met_id) & target_key_set:
                return True
        if filter_currency and _is_currency(met_id):
            return False
        return True

    # Build full-GEM compound-to-reaction incidence; currency metabolites are
    # omitted from edges but remain available as indexed compound nodes.
    reaction_compounds = {}
    reaction_genes = {}
    compound_reaction_edges = []
    reaction_gene_edges = []

    filtered_currency = 0
    unmatched_compounds = 0
    for r_idx, r in enumerate(reactions):
        c_indices = set()
        for met_id, _ in r['metabolites'].items():
            if not should_include_compound(met_id):
                filtered_currency += 1
                continue
            matching_keys = _metabolite_id_keys(met_id) & set(compound_key_to_idx)
            if not matching_keys:
                unmatched_compounds += 1
                continue
            ci = compound_key_to_idx[sorted(matching_keys)[0]]
            c_indices.add(ci)
        reaction_compounds[r_idx] = sorted(c_indices)
        for ci in reaction_compounds[r_idx]:
            compound_reaction_edges.append((ci, r_idx))

        grr = r['gene_reaction_rule']
        gene_matches = re.findall(r'b\d{4}|s\d{4}', grr, re.I)
        if not gene_matches:
            gene_matches = re.findall(r'[A-Za-z0-9_]{3,}', grr)
            gene_matches = [x for x in gene_matches if x.lower() not in ('and', 'or')]
        for g in gene_matches:
            if g in gene_to_idx:
                gi = gene_to_idx[g]
                reaction_gene_edges.append((r_idx, gi))
                if r_idx not in reaction_genes:
                    reaction_genes[r_idx] = []
                reaction_genes[r_idx].append(gi)

    # Incidence matrix: (num_compounds, num_reactions)
    num_compounds = len(compound_names)
    num_reactions = len(reactions)
    incidence = torch.zeros(num_compounds, num_reactions)
    for r_idx, c_list in reaction_compounds.items():
        for ci in c_list:
            incidence[ci, r_idx] = 1.0
    nonzero_compounds = int((incidence.sum(dim=1) > 0).sum().item())
    active_reactions = int((incidence.sum(dim=0) > 0).sum().item())
    isolated_targets = sum(
        1 for target_name in target_metabolites
        if target_name in target_to_compound_idx and incidence[target_to_compound_idx[target_name]].sum().item() == 0
    )
    missing_target_nodes = len(target_metabolites) - len(target_to_compound_idx)
    smiles_coverage = sum(1 for smiles in metabolite_smiles if smiles.strip())
    print(
        "Full-GEM hypergraph: "
        f"compound_nodes={num_compounds}, target_nodes={len(target_metabolites)}, "
        f"connected_nodes={nonzero_compounds}, isolated_targets={isolated_targets}, "
        f"missing_target_nodes={missing_target_nodes}, active_reactions={active_reactions}, "
        f"compound_edges={len(compound_reaction_edges)}, "
        f"smiles_coverage={smiles_coverage}/{num_compounds}, "
        f"filtered_currency_entries={filtered_currency}, unmatched_entries={unmatched_compounds}"
    )

    return (
        gene_sequences, metabolite_smiles, task_labels, gene_names, metabolite_names,
        reaction_compounds, reaction_genes, compound_reaction_edges, reaction_gene_edges,
        gpr_rules, gene_to_idx, incidence, num_compounds, num_reactions,
        compound_names, target_to_compound_idx
    )


def load_metabolic_data(data_path: str, cbm: Optional[str] = None) -> Tuple:
    """
    Load metabolic data. With ``cbm``, resolves ``data_path/Data/<cbm>/``.
    Without ``cbm``, treats ``data_path`` as a GEM directory and returns a shorter tuple.
    """
    if cbm:
        return load_metabolic_data_for_gem(data_path, cbm)
    # data_path is the GEM directory itself
    base = os.path.dirname(data_path)
    cbm_name = os.path.basename(data_path)
    result = load_metabolic_data_for_gem(base, cbm_name)
    # Omit incidence matrix and GEM size fields when cbm is omitted
    return result[:11]  # Exclude incidence, num_compounds, num_reactions


def create_metabolite_based_cross_validation(dataset, num_folds: int = 5, seed: int = 42):
    """Split by metabolites for cross-validation. Yields (train_idx, val_idx, train_metas, val_metas)."""
    import random
    random.seed(seed)
    metas = list(set(m for _, m in dataset.pairs))
    random.shuffle(metas)
    n = len(metas)
    fold_size = max(1, n // num_folds)
    for f in range(num_folds):
        start = f * fold_size
        end = start + fold_size if f < num_folds - 1 else n
        val_metas = set(metas[start:end])
        train_metas = set(metas) - val_metas
        train_idx = [i for i, (_, m) in enumerate(dataset.pairs) if m in train_metas]
        val_idx = [i for i, (_, m) in enumerate(dataset.pairs) if m in val_metas]
        if train_idx and val_idx:
            yield train_idx, val_idx, list(train_metas), list(val_metas)


def filter_edges_for_metabolite_split(compound_reaction_edges, reaction_gene_edges,
                                       train_metabolites, val_metabolites, metabolite_names,
                                       reaction_compounds, reaction_genes):
    """Keep edges for training metabolites; include reaction-gene if reaction involves train metabolites."""
    train_meta_set = set(train_metabolites)
    meta_to_idx = {m: i for i, m in enumerate(metabolite_names)}
    train_indices = {meta_to_idx[m] for m in train_meta_set if m in meta_to_idx}
    reactions_with_train = set()
    for ci, ri in compound_reaction_edges:
        if ci in train_indices:
            reactions_with_train.add(ri)
    cr_edges = [(c, r) for c, r in compound_reaction_edges if c in train_indices]
    rg_edges = [(r, g) for r, g in reaction_gene_edges if r in reactions_with_train]
    return cr_edges, rg_edges


def build_fold_structure(
    train_metabolites: List[str],
    metabolite_names: List[str],
    reaction_compounds: Dict[int, List[int]],
    reaction_genes: Dict[int, List[int]],
    compound_reaction_edges: List[Tuple[int, int]],
    reaction_gene_edges: List[Tuple[int, int]],
    gpr_rules: List[str],
    incidence_matrix: torch.Tensor,
    target_to_compound_idx: Optional[Dict[str, int]] = None,
) -> Dict:
    """Build the train-only metabolic structure for one metabolite CV fold.

    The tensor shape is kept identical to the full hypergraph so cached
    metabolite indices remain valid. In full-GEM mode, non-target compound
    nodes remain available while validation target metabolite rows are masked
    out; in target-only mode, only training target rows are kept.
    """
    if target_to_compound_idx is None:
        meta_to_idx = {m: i for i, m in enumerate(metabolite_names)}
        train_indices = {meta_to_idx[m] for m in train_metabolites if m in meta_to_idx}
    else:
        train_target_indices = {
            target_to_compound_idx[m]
            for m in train_metabolites
            if m in target_to_compound_idx
        }
        all_target_indices = set(target_to_compound_idx.values())
        validation_target_indices = all_target_indices - train_target_indices
        all_compound_indices = set(range(incidence_matrix.size(0)))
        train_indices = all_compound_indices - validation_target_indices

    train_incidence = torch.zeros_like(incidence_matrix)
    train_reaction_compounds = {}
    train_compound_reaction_edges = []
    reactions_with_train = set()

    for r_idx, compounds in reaction_compounds.items():
        kept_compounds = sorted({ci for ci in compounds if ci in train_indices})
        train_reaction_compounds[r_idx] = kept_compounds
        if kept_compounds:
            reactions_with_train.add(r_idx)
            for ci in kept_compounds:
                train_incidence[ci, r_idx] = 1.0
                train_compound_reaction_edges.append((ci, r_idx))

    train_reaction_genes = {
        r_idx: list(genes)
        for r_idx, genes in reaction_genes.items()
        if r_idx in reactions_with_train
    }
    train_reaction_gene_edges = [
        (r_idx, gene_idx)
        for r_idx, gene_idx in reaction_gene_edges
        if r_idx in reactions_with_train
    ]
    train_gpr_rules = [
        rule if r_idx in reactions_with_train else ""
        for r_idx, rule in enumerate(gpr_rules or [])
    ]

    return {
        "incidence_matrix": train_incidence,
        "reaction_compounds": train_reaction_compounds,
        "reaction_genes": train_reaction_genes,
        "compound_reaction_edges": train_compound_reaction_edges,
        "reaction_gene_edges": train_reaction_gene_edges,
        "gpr_rules": train_gpr_rules,
        "train_metabolite_indices": sorted(train_indices),
        "active_reaction_indices": sorted(reactions_with_train),
    }


def write_metabolite_gene_predictions_csv(model, gene_sequences, metabolite_smiles, gene_names, metabolite_names,
                                   task_labels, reaction_compounds, reaction_genes, compound_reaction_edges,
                                   reaction_gene_edges, data_path, device, gpr_rules=None, gene_to_idx=None,
                                   incidence_matrix=None, output_dir=None,
                                   gene_backbone_features=None, metabolite_backbone_features=None,
                                   target_to_compound_idx=None,
                                   binary_threshold: BinaryThresholdMode = DEFAULT_BINARY_THRESHOLD_MODE,
                                   train_metabolites: Optional[Iterable[str]] = None):
    """Write long-form ``all_metabolites_predictions.csv`` (gene--metabolite rows with scores).
    If incidence_matrix is provided, model is assumed to be HGPRGDelHypergraphModel (full data + batch indices).
    If output_dir is provided, saves to that directory instead of data_path/Results.
    """
    all_pairs = [(g, m) for g in gene_names for m in metabolite_names]
    predictions: list[float] = []

    model.eval()
    with torch.no_grad():
        if incidence_matrix is not None and hasattr(model, 'hypergraph'):
            from model.gpr_module import (
                infer_num_reactions,
                parse_gpr_rules_list,
                pad_gpr_parsed,
                build_gene_compound_reaction_incidence,
            )
            gpr_parsed_kw = {}
            if getattr(model, 'use_gpr', True):
                num_rxn_inf = infer_num_reactions(reaction_compounds, reaction_genes, gpr_rules)
                gpr_parsed_kw['gpr_parsed'] = pad_gpr_parsed(
                    parse_gpr_rules_list(gpr_rules, model.gpr_parser),
                    num_rxn_inf,
                )
                g_rxn_inf, c_rxn_inf = build_gene_compound_reaction_incidence(
                    reaction_genes or {},
                    reaction_compounds or {},
                    len(gene_names),
                    len(metabolite_smiles),
                    num_rxn_inf,
                    device,
                )
                gpr_parsed_kw['gene_rxn_incidence'] = g_rxn_inf
                gpr_parsed_kw['compound_rxn_incidence'] = c_rxn_inf
            g_idx = [gene_names.index(g) for g, _ in all_pairs]
            if target_to_compound_idx is None:
                m_idx = [metabolite_names.index(m) for _, m in all_pairs]
            else:
                m_idx = [target_to_compound_idx[m] for _, m in all_pairs]
            batch_size = 64
            for i in range(0, len(all_pairs), batch_size):
                end = min(i + batch_size, len(all_pairs))
                pred = model(gene_sequences, metabolite_smiles, reaction_compounds, reaction_genes,
                            compound_reaction_edges, reaction_gene_edges,
                            incidence_matrix=incidence_matrix.to(device), gpr_rules=gpr_rules,
                            gene_to_idx=gene_to_idx, batch_gene_idx=g_idx[i:end], batch_metabolite_idx=m_idx[i:end],
                            gene_backbone_features=gene_backbone_features,
                            metabolite_backbone_features=metabolite_backbone_features,
                            **gpr_parsed_kw)
                for v in torch.sigmoid(pred).cpu().numpy():
                    predictions.append(float(v))
        else:
            for gene_name, metabolite_name in all_pairs:
                gi = gene_names.index(gene_name) if gene_name in gene_names else 0
                mi = metabolite_names.index(metabolite_name) if metabolite_name in metabolite_names else 0
                g_seq = gene_sequences[gi] if gi < len(gene_sequences) else gene_sequences[0]
                m_smi = metabolite_smiles[mi] if mi < len(metabolite_smiles) else metabolite_smiles[0]
                pred = model([g_seq], [m_smi], reaction_compounds, reaction_genes,
                            compound_reaction_edges, reaction_gene_edges,
                            gpr_rules=gpr_rules, gene_to_idx=gene_to_idx)
                v = torch.sigmoid(pred).item() if pred.numel() == 1 else torch.sigmoid(pred)[0].item()
                predictions.append(v)

    binary_preds = binary_predictions_for_pairs(
        all_pairs,
        predictions,
        task_labels,
        mode=binary_threshold,
        train_metabolites=train_metabolites,
    )

    rows = [{'Gene': g, 'Metabolite': m, 'Prediction': predictions[i], 'Binary_Prediction': binary_preds[i],
             'True_Label': task_labels.get((g, m), -1)} for i, (g, m) in enumerate(all_pairs)]
    df = pd.DataFrame(rows)
    out_dir = output_dir if output_dir is not None else os.path.join(data_path, "Results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "all_metabolites_predictions.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file}")
    return df, binary_preds, predictions
