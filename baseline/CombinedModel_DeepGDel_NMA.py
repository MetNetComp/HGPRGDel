"""
NMA (Neighborhood Mean Aggregation) baseline model.
Same as DeepGDel (Gene-M + Meta-M) but prediction uses NMA-aggregated metabolite features
instead of the metabolite's own Meta-M output.
"""
import torch
import torch.nn as nn


class CombinedModel_DeepGDel_NMA(nn.Module):
    """
    DeepGDel with NMA: Gene-M and Meta-M identical to DeepGDel; prediction branch uses
    Z_meta_nma (neighborhood mean of Meta-M outputs) instead of per-sample Meta-M output.
    """

    def __init__(self, gene_lstm, smiles_model, hidden_dim):
        super(CombinedModel_DeepGDel_NMA, self).__init__()
        self.gene_lstm = gene_lstm
        self.smiles_model = smiles_model
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, gene_seq, smiles_seq, metabolite_indices, Z_meta_nma):
        """
        Args:
            gene_seq: (batch_size, seq_length)
            smiles_seq: (batch_size, smiles_seq_length) for reconstruction
            metabolite_indices: (batch_size,) long, index into Z_meta_nma
            Z_meta_nma: (num_nodes, hidden_dim) precomputed NMA features

        Returns:
            output: (batch_size, 1) classification logits (sigmoid applied in loss)
            gene_recon, smiles_recon: for reconstruction loss
        """
        gene_embedding, gene_recon = self.gene_lstm(gene_seq)
        smiles_embedding, smiles_recon = self.smiles_model(smiles_seq)

        meta_for_pred = Z_meta_nma[metabolite_indices]
        combined_feat = gene_embedding * meta_for_pred
        output = torch.sigmoid(self.fc(combined_feat))
        return output, gene_recon, smiles_recon
