"""
Dataset for NMA baseline: same as DeepGDel but each sample includes metabolite_index
for indexing into Z_meta_nma.
"""
import torch
from torch.utils.data import Dataset


def create_gene_metabolite_dataset_nma(gene_sequences, smiles_features, labels, metabolite_indices):
    """
    Create a Dataset that returns (gene_seq, smiles_feat, metabolite_index, label)
    for NMA baseline.

    Parameters:
        gene_sequences: list/array of padded tokenized gene sequences.
        smiles_features: list/array of padded tokenized SMILES sequences.
        labels: list/array of labels.
        metabolite_indices: list/array of node indices (same length as labels).

    Returns:
        Dataset yielding (gene_seq, smiles_seq, metabolite_index, label) per sample.
    """
    class GeneMetaboliteDatasetNMA(Dataset):
        def __init__(self, gene_sequences, smiles_features, labels, metabolite_indices):
            self.gene_sequences = gene_sequences
            self.smiles_features = smiles_features
            self.labels = labels
            self.metabolite_indices = metabolite_indices

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            gene_seq = self.gene_sequences[idx]
            smiles_feat = self.smiles_features[idx]
            label = self.labels[idx]
            meta_idx = self.metabolite_indices[idx]
            return (
                torch.tensor(gene_seq, dtype=torch.long),
                torch.tensor(smiles_feat, dtype=torch.long),
                meta_idx,
                torch.tensor(label, dtype=torch.float),
            )

    return GeneMetaboliteDatasetNMA(
        gene_sequences, smiles_features, labels, metabolite_indices
    )
