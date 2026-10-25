import torch
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def create_gene_metabolite_dataset(
    metabolites,
    relationships,
    gene_sequences,
    smiles_features,
    filtered_genes,
    graph_data,
    device,
    gene_names=None,
    metabolite_names=None,
    graph_node_index_map=None,
):
    """
    Create a dataset for gene sequences, SMILES features, and graph data.
    Data is already tokenized.

    When graph_node_index_map is provided, node_index = graph_node_index_map[metabolite]
    and gene/smiles sequences are looked up by gene_names and metabolite_names.
    When not provided, fallback: node_index = hash(metabolite) % num_nodes,
    gene_seq/smiles_seq = first row (placeholder).
    """
    if gene_names is None:
        gene_names = []
    if metabolite_names is None:
        metabolite_names = []
    metabolite_names_arr = np.asarray(metabolite_names) if metabolite_names else np.array([])

    class GeneMetaboliteGraphDataset(Dataset):
        def __init__(
            self,
            metabolites,
            relationships,
            gene_sequences,
            smiles_features,
            filtered_genes,
            graph_data,
            device,
            gene_names,
            metabolite_names_arr,
            graph_node_index_map,
        ):
            self.metabolites = metabolites
            self.relationships = relationships
            self.gene_sequences = gene_sequences
            self.smiles_features = smiles_features
            self.filtered_genes = filtered_genes
            self.graph_data = graph_data
            self.device = device
            self.gene_names = list(gene_names) if gene_names is not None and len(gene_names) else []
            self.metabolite_names_arr = metabolite_names_arr
            self.graph_node_index_map = graph_node_index_map

            self.samples = []
            for metabolite in metabolites:
                if metabolite in relationships:
                    gene_names_list, labels = relationships[metabolite]
                    for gene_name, label in zip(gene_names_list, labels):
                        if gene_name in filtered_genes:
                            self.samples.append((metabolite, gene_name, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            metabolite, gene_name, label = self.samples[idx]

            if self.gene_names and len(self.gene_sequences) > 0:
                try:
                    gi = self.gene_names.index(gene_name)
                    gene_seq = self.gene_sequences[gi]
                except (ValueError, IndexError):
                    gene_seq = self.gene_sequences[0]
            else:
                gene_seq = self.gene_sequences[0] if len(self.gene_sequences) > 0 else np.zeros(100)

            if self.metabolite_names_arr.size > 0 and len(self.smiles_features) > 0:
                try:
                    mi = np.where(self.metabolite_names_arr == metabolite)[0][0]
                    smiles_seq = self.smiles_features[mi]
                except (IndexError, TypeError):
                    smiles_seq = self.smiles_features[0] if len(self.smiles_features) > 0 else np.zeros(100)
            else:
                smiles_seq = self.smiles_features[0] if len(self.smiles_features) > 0 else np.zeros(100)

            if self.graph_node_index_map is not None:
                node_index = self.graph_node_index_map.get(metabolite, -1)
            else:
                node_index = hash(metabolite) % self.graph_data.num_nodes

            return (
                torch.tensor(gene_seq, dtype=torch.long),
                torch.tensor(smiles_seq, dtype=torch.long),
                torch.tensor(label, dtype=torch.float),
                torch.tensor(node_index, dtype=torch.long),
                self.graph_data,
            )

    return GeneMetaboliteGraphDataset(
        metabolites,
        relationships,
        gene_sequences,
        smiles_features,
        filtered_genes,
        graph_data,
        device,
        gene_names,
        metabolite_names_arr,
        graph_node_index_map,
    )
