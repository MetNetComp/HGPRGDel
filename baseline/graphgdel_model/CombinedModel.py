import torch
import torch.nn as nn 

class CombinedModel(nn.Module):
    """
    CombinedModel: Integrates GeneLSTM, SMILESLSTM, and GCN for joint feature learning.
    
    Args:
        gene_lstm (nn.Module): GeneLSTM model.
        smiles_model (nn.Module): SMILESLSTM model.
        gcn_model (nn.Module): GCN model.
        hidden_dim (int): Dimensionality of the feature space.
    """
    def __init__(self, gene_lstm, smiles_model, gcn_model, hidden_dim):
        super(CombinedModel, self).__init__()
        self.gene_lstm = gene_lstm
        self.smiles_model = smiles_model
        self.gcn_model = gcn_model
        self.fc = nn.Linear(hidden_dim * 3, 1)  # gene + smiles + gcn = 3 * hidden_dim

    def forward(self, gene_seq, smiles_seq, graph, gcp_node_indices):
        """
        Forward pass for the combined model.
        
        Args:
            gene_seq (Tensor): Input gene sequences (batch_size, seq_length).
            smiles_seq (Tensor): Input SMILES sequences (batch_size, seq_length).
            graph (Data): PyTorch Geometric Data object containing graph structure.
            gcp_node_indices (Tensor): Node indices for GCN feature selection.
        
        Returns:
            output (Tensor): Binary classification output (batch_size, 1).
            reconstructed_gene (Tensor): Reconstructed gene sequences.
            reconstructed_smiles (Tensor): Reconstructed SMILES sequences.
        """
        gene_embedding, gene_recon = self.gene_lstm(gene_seq)
        smiles_embedding, smiles_recon = self.smiles_model(smiles_seq)
        
        gcn_embeddings = self.gcn_model.encode(graph.x, graph.edge_index)
        gcn_selected = gcn_embeddings[gcp_node_indices]

        # Use the same feature combination as source code
        # The source code uses concatenation of gene_embedding and gcn_selected
        # But the saved model expects 192 features, which suggests it might be using
        # gene_embedding (64) + smiles_embedding (64) + gcn_selected (64) = 192
        combined_feat = torch.cat([gene_embedding, smiles_embedding, gcn_selected], dim=-1)
        
        # Classification output
        output = torch.sigmoid(self.fc(combined_feat))
        return output, gene_recon, smiles_recon
