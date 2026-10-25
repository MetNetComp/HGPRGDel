import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, gene_vocab_size, smiles_vocab_size, max_gene_len, max_smiles_len, embed_dim=32):
        super(DNN, self).__init__()
        self.embed_dim = embed_dim

        self.gene_embedding = nn.Embedding(gene_vocab_size, embed_dim)
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, embed_dim)

        input_dim = (max_gene_len + max_smiles_len) * embed_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, gene_input, smiles_input):
        gene_embed = self.gene_embedding(gene_input.long())       # (B, max_gene_len, E)
        smiles_embed = self.smiles_embedding(smiles_input.long()) # (B, max_smiles_len, E)

        gene_embed = gene_embed.view(gene_embed.size(0), -1)
        smiles_embed = smiles_embed.view(smiles_embed.size(0), -1)

        combined_feat = torch.cat([gene_embed, smiles_embed], dim=-1)

        return torch.sigmoid(self.fc(combined_feat))