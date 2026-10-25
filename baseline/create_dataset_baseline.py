import torch
from torch.utils.data import Dataset

def create_gene_metabolite_dataset(gene_sequences, smiles_features, labels):
    """
    Function to create a dataset for gene sequences and SMILES features for baseline methods.
    This is a simplified version that doesn't require graph data.

    Parameters:
    - gene_sequences (list): List of gene sequences (as numeric values).
    - smiles_features (list): List of SMILES features (as numeric values).
    - labels (list): List of labels corresponding to the sequences.

    Returns:
    - dataset (Dataset): A PyTorch Dataset containing the gene sequences, SMILES features, and labels.
    """

    class GeneMetaboliteDataset(Dataset):
        def __init__(self, gene_sequences, smiles_features, labels):
            self.gene_sequences = gene_sequences
            self.smiles_features = smiles_features
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            gene_seq = self.gene_sequences[idx]
            smiles_feat = self.smiles_features[idx]
            label = self.labels[idx]
            return torch.tensor(gene_seq, dtype=torch.float), torch.tensor(smiles_feat, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    # Return the dataset
    return GeneMetaboliteDataset(gene_sequences, smiles_features, labels)
