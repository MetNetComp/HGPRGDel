"""
HGPRGDel Learning Modules
==========================

This package contains the core learning modules for HGPRGDel:
- Gene Learning Module: Protein sequence understanding using ESM-2
- Metabolite Learning Module: Chemical structure understanding using ChemBERTa

The learning modules are designed to be stable and reusable across different downstream tasks.
"""

try:
    from .gene_module import ESM2GeneEncoder, GeneLearningModule
except Exception:
    ESM2GeneEncoder = None
    GeneLearningModule = None

try:
    from .metabolite_module import MolBERTMetaboliteEncoder, MetaboliteLearningModule
except Exception:
    MolBERTMetaboliteEncoder = None
    MetaboliteLearningModule = None
from .gpr_module import GPRRuleParser, SoftBooleanAggregator, MeanPoolingGPRAggregator
from utils.data_module import (
    MetabolicDataset, BalancedMetabolicDataset, load_metabolic_data, load_metabolic_data_for_gem,
    write_metabolite_gene_predictions_csv, get_evaluation_genes, CURRENCY_METABOLITES,
    create_metabolite_based_cross_validation, filter_edges_for_metabolite_split, build_fold_structure,
)
from .hypergraph_module import HypergraphCompoundEncoder
try:
    from .hgprgdel_model import HGPRGDelHypergraphModel
except Exception:
    HGPRGDelHypergraphModel = None
from trainer.training_module import train_epoch, validate_epoch, validate_epoch_with_metabolite_thresholds, evaluate_model

__all__ = [
    'ESM2GeneEncoder', 'MolBERTMetaboliteEncoder',
    'GPRRuleParser', 'SoftBooleanAggregator', 'MeanPoolingGPRAggregator',
    'MetabolicDataset', 'BalancedMetabolicDataset',
    'load_metabolic_data', 'load_metabolic_data_for_gem',
    'write_metabolite_gene_predictions_csv', 'get_evaluation_genes',
    'CURRENCY_METABOLITES', 'create_metabolite_based_cross_validation',
    'filter_edges_for_metabolite_split', 'build_fold_structure',
    'HypergraphCompoundEncoder', 'HGPRGDelHypergraphModel',
    'train_epoch', 'validate_epoch', 'validate_epoch_with_metabolite_thresholds',
    'evaluate_model',
    'GeneLearningModule', 'MetaboliteLearningModule',
]
