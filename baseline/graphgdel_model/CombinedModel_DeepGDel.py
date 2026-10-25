import torch
import torch.nn as nn 

class CombinedModel_DeepGDel(nn.Module):
    """
    CombinedModel for DeepGDel: Integrates GeneLSTM and SMILESLSTM for joint feature learning.
    
    Args:
        gene_lstm (nn.Module): GeneLSTM model.
        smiles_model (nn.Module): SMILESLSTM model.
        hidden_dim (int): Dimensionality of the feature space.
    """
    def __init__(self, gene_lstm, smiles_model, hidden_dim):
        super(CombinedModel_DeepGDel, self).__init__()
        self.gene_lstm = gene_lstm
        self.smiles_model = smiles_model
        self.fc = nn.Linear(hidden_dim, 1)  # Use hidden_dim to match reference

    def forward(self, gene_seq, smiles_seq):
        """
        Forward pass for the combined model.
        
        Args:
            gene_seq (Tensor): Input gene sequences (batch_size, seq_length).
            smiles_seq (Tensor): Input SMILES sequences (batch_size, seq_length).
        
        Returns:
            output (Tensor): Binary classification output (batch_size, 1).
            reconstructed_gene (Tensor): Reconstructed gene sequences.
            reconstructed_smiles (Tensor): Reconstructed SMILES sequences.
        """
        gene_embedding, gene_recon = self.gene_lstm(gene_seq)
        smiles_embedding, smiles_recon = self.smiles_model(smiles_seq)
        
        # Use element-wise multiplication like reference
        combined_feat = gene_embedding * smiles_embedding
        output = torch.sigmoid(self.fc(combined_feat))
        return output, gene_recon, smiles_recon
