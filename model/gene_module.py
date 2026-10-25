"""
HGPRGDel Gene Learning Module
=============================

This module uses ESM-2 (Evolutionary Scale Modeling) protein language model
to learn latent features from amino acid sequences of gene products.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import numpy as np


class ESM2GeneEncoder(nn.Module):
    """
    ESM-2 based gene encoder for amino acid sequences.
    
    Uses ESM-2 model as a frozen feature extractor to provide 
    pre-trained protein embeddings for amino acid sequences.
    """
    
    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",  # Already the smallest ESM-2 model
        hidden_size: int = 128,  # Project to 128D
        dropout: float = 0.0,
        freeze_backbone: bool = True  # Always freeze for feature extraction
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # Load ESM-2 model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        self.esm_model = AutoModel.from_pretrained(f"facebook/{model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(self.esm_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Sequence reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, self.tokenizer.vocab_size)
        )
        
        # Pooling strategy
        self.pooling = "mean"  # Options: mean, max, cls
        
    def forward(
        self,
        sequences: List[str],
        return_features: bool = True,
        return_reconstruction: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the gene encoder.
        
        Args:
            sequences: List of amino acid sequences
            return_features: Whether to return protein features
            return_reconstruction: Whether to return reconstruction logits
            
        Returns:
            Dictionary containing features and/or reconstruction logits
        """
        
        pooled_features = self.extract_pooled_features(sequences)
        protein_features = self.project_pooled_features(pooled_features)
        
        result = {}
        
        if return_features:
            result["protein_features"] = protein_features
            
        if return_reconstruction:
            # Generate reconstruction logits
            reconstruction_logits = self.reconstruction_head(protein_features)
            result["reconstruction_logits"] = reconstruction_logits
            
        return result

    def extract_pooled_features(self, sequences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Return frozen ESM pooled hidden states before the trainable projection."""
        pooled_batches = []
        device = next(self.esm_model.parameters()).device
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start:start + batch_size]
            batch_encoding = self.tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
            with torch.no_grad():
                outputs = self.esm_model(**batch_encoding)
                hidden_states = outputs.last_hidden_state
            attention_mask = batch_encoding["attention_mask"]
            if self.pooling == "mean":
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            elif self.pooling == "max":
                masked_hidden = hidden_states.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
                pooled = masked_hidden.max(dim=1)[0]
            else:
                pooled = hidden_states[:, 0, :]
            pooled_batches.append(pooled.detach().cpu())
        return torch.cat(pooled_batches, dim=0)

    def project_pooled_features(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Project cached ESM pooled hidden states to the trainable gene feature space."""
        device = next(self.feature_projection.parameters()).device
        return self.feature_projection(pooled_features.to(device))
    
    def get_sequence_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """
        Get protein embeddings for a list of sequences.
        
        Args:
            sequences: List of amino acid sequences
            
        Returns:
            Protein embeddings tensor [batch_size, hidden_size]
        """
        with torch.no_grad():
            result = self.forward(sequences, return_features=True, return_reconstruction=False)
            return result["protein_features"]


class GeneLearningModule(nn.Module):
    """
    Complete gene learning module with training capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        hidden_size: int = 320,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        reconstruction_weight: float = 1.0
    ):
        super().__init__()
        
        self.encoder = ESM2GeneEncoder(
            model_name=model_name,
            hidden_size=hidden_size,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        self.reconstruction_weight = reconstruction_weight
        
    def forward(
        self,
        sequences: List[str],
        targets: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional reconstruction loss.
        
        Args:
            sequences: Input amino acid sequences
            targets: Target sequences for reconstruction (optional)
            
        Returns:
            Dictionary with features, loss, and metrics
        """
        
        # Get encoder outputs
        encoder_outputs = self.encoder(
            sequences,
            return_features=True,
            return_reconstruction=targets is not None
        )
        
        result = {
            "protein_features": encoder_outputs["protein_features"]
        }
        
        # Calculate reconstruction loss if targets provided
        if targets is not None and "reconstruction_logits" in encoder_outputs:
            loss = self._compute_reconstruction_loss(
                encoder_outputs["reconstruction_logits"],
                targets,
                self.encoder.tokenizer
            )
            result["loss"] = loss
            result["reconstruction_loss"] = loss
            
        return result
    
    def _compute_reconstruction_loss(
        self,
        logits: torch.Tensor,
        targets: List[str],
        tokenizer
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for sequence prediction.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            targets: Target sequences
            tokenizer: ESM tokenizer
            
        Returns:
            Reconstruction loss
        """
        # Tokenize targets
        target_encoding = tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        device = logits.device
        target_ids = target_encoding["input_ids"].to(device)
        
        # Ensure logits and targets have compatible shapes
        # Logits: [batch_size, vocab_size]
        # Target IDs: [batch_size, seq_len]
        
        # For reconstruction, we'll use the first token as target (simplified approach)
        # In practice, you might want to use a more sophisticated approach
        target_token = target_ids[:, 0]  # Take first token of each sequence
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits,  # [batch_size, vocab_size]
            target_token,  # [batch_size]
            ignore_index=tokenizer.pad_token_id
        )
        
        return loss * self.reconstruction_weight
    
    def get_features(self, sequences: List[str]) -> torch.Tensor:
        """
        Extract protein features for downstream tasks.
        
        Args:
            sequences: Amino acid sequences
            
        Returns:
            Protein feature embeddings
        """
        return self.encoder.get_sequence_embeddings(sequences)


def load_pretrained_gene_model(model_path: str) -> GeneLearningModule:
    """
    Load a pretrained gene learning module.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded gene learning module
    """
    model = GeneLearningModule()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model


def save_gene_model(model: GeneLearningModule, model_path: str):
    """
    Save a trained gene learning module.
    
    Args:
        model: Gene learning module to save
        model_path: Path to save the model
    """
    torch.save(model.state_dict(), model_path)
