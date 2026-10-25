import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

class GCN(nn.Module):
    """
    Graph Convolutional Network for metabolic network processing.
    This module is responsible for learning graph-structured representations.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes, pos_emb_dim=16):
        super(GCN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        
        # Graph convolutional layers (matching source code)
        self.conv1 = GCNConv(in_channels + pos_emb_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Positional embedding (matching source code)
        self.pos_embedding = nn.Embedding(num_nodes, pos_emb_dim)
        
        # Link prediction layers (matching source code)
        self.link_pred = nn.Linear(out_channels * 2, 1)
        
    def encode(self, x, edge_index):
        """
        Encode the graph structure into node embeddings.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Get node indices for positional embedding
        node_indices = torch.arange(x.size(0), device=x.device)
        
        # Get positional embeddings
        pos_emb = self.pos_embedding(node_indices)
        
        # Concatenate node features with positional embeddings
        x = torch.cat([x, pos_emb], dim=1)
        
        # Graph convolutions (matching source code)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        
        return x
    
    def decode(self, z, edge_label_index):
        """
        Decode node embeddings to predict edge existence.
        
        Args:
            z: Node embeddings [num_nodes, out_channels]
            edge_label_index: Edge indices for prediction [2, num_edges]
            
        Returns:
            Edge predictions [num_edges]
        """
        # Get source and target node embeddings
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        
        # Concatenate source and target embeddings
        z_concat = torch.cat([z_src, z_dst], dim=1)
        
        # Predict edge existence
        edge_pred = self.link_pred(z_concat)
        
        return edge_pred.squeeze(-1)
    
    def forward(self, x, edge_index, edge_label_index=None):
        """
        Forward pass through the GCN model.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_label_index: Edge indices for link prediction [2, num_edges] (optional)
            
        Returns:
            Node embeddings [num_nodes, out_channels]
            Edge predictions [num_edges] (if edge_label_index is provided)
        """
        z = self.encode(x, edge_index)
        
        if edge_label_index is not None:
            edge_pred = self.decode(z, edge_label_index)
            return z, edge_pred
        
        return z
