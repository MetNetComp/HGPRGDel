"""
HGPRGDel Metabolite Learning Module
====================================

Uses ChemBERTa (SMILES-specific) or DistilBERT fallback for metabolite encoding.
Supports optional fine-tuning via freeze_backbone=False.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import numpy as np

Chem = AllChem = None


def _load_rdkit():
    """Lazily load optional RDKit modules only when similarity features are requested."""
    global Chem, AllChem
    if Chem is not None and AllChem is not None:
        return Chem, AllChem
    try:
        from rdkit import Chem as _Chem
        from rdkit.Chem import AllChem as _AllChem
    except Exception as exc:
        raise RuntimeError("RDKit similarity features are unavailable in this environment") from exc
    Chem, AllChem = _Chem, _AllChem
    return Chem, AllChem


def get_metabolite_encoder(use_chemberta: bool = True, model_name: str = None, **kwargs) -> nn.Module:
    """Factory: ChemBERTa if available, else MolBERT (DistilBERT)."""
    if use_chemberta:
        try:
            name = model_name or "DeepChem/ChemBERTa-77M-MLM"
            return ChemBERTaMetaboliteEncoder(model_name=name, **kwargs)
        except Exception as e:
            err = str(e)
            if "torch.load" in err or "CVE" in err or "2.6" in err:
                print("ChemBERTa load failed (CVE-2025-32434 requires PyTorch>=2.6). "
                      "Upgrade: pip install 'torch>=2.6'. Using MolBERT fallback.")
            else:
                print(f"ChemBERTa load failed ({e}), using MolBERT fallback")
    return MolBERTMetaboliteEncoder(model_name=model_name or "distilbert-base-uncased", **kwargs)


class ChemBERTaMetaboliteEncoder(nn.Module):
    """
    ChemBERTa encoder for SMILES - pre-trained on chemical data.
    Model: DeepChem/ChemBERTa-77M-MLM
    """
    def __init__(self, model_name: str = "DeepChem/ChemBERTa-77M-MLM", hidden_size: int = 256,
                 dropout: float = 0.0, freeze_backbone: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
        dim = self.model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.recon_head = nn.Linear(hidden_size, self.tokenizer.vocab_size)
        self.hidden_size = hidden_size

    def forward(self, smiles_list: List[str], return_features: bool = True, return_reconstruction: bool = False):
        h = self.extract_pooled_features(smiles_list)
        feat = self.project_pooled_features(h)
        out_dict = {}
        if return_features:
            out_dict["chemical_features"] = feat
        if return_reconstruction:
            out_dict["reconstruction_logits"] = self.recon_head(feat)
        return out_dict

    def extract_pooled_features(self, smiles_list: List[str], batch_size: int = 64) -> torch.Tensor:
        """Return frozen ChemBERTa pooled hidden states before the trainable projection."""
        pooled_batches = []
        device = next(self.model.parameters()).device
        for start in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[start:start + batch_size]
            enc = self.tokenizer(batch_smiles, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
            mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"], dtype=torch.float))
            h = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            pooled_batches.append(h.detach().cpu())
        return torch.cat(pooled_batches, dim=0)

    def project_pooled_features(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """Project cached ChemBERTa pooled hidden states to the metabolite feature space."""
        device = next(self.proj.parameters()).device
        return self.proj(pooled_features.to(device))


class MolBERTMetaboliteEncoder(nn.Module):
    """
    DistilBERT-based metabolite encoder (fallback when ChemBERTa unavailable).
    
    Uses DistilBERT model (smaller, faster, PyTorch 2.2.2 compatible) as a 
    frozen feature extractor to provide pre-trained embeddings for SMILES sequences.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",  # Smaller, faster model
        hidden_size: int = 128,  # Project to 128D
        dropout: float = 0.0,
        freeze_backbone: bool = True  # Always freeze for feature extraction
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # Load DistilBERT model and tokenizer (smaller, faster, PyTorch 2.2.2 compatible)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.distilbert_model = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone for feature extraction (no training)
        if freeze_backbone:
            for param in self.distilbert_model.parameters():
                param.requires_grad = False
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(self.distilbert_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # SMILES reconstruction head
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
        smiles_list: List[str],
        return_features: bool = True,
        return_reconstruction: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the metabolite encoder.
        
        Args:
            smiles_list: List of SMILES strings
            return_features: Whether to return chemical features
            return_reconstruction: Whether to return reconstruction logits
            
        Returns:
            Dictionary containing features and/or reconstruction logits
        """
        
        pooled_features = self.extract_pooled_features(smiles_list)
        chemical_features = self.project_pooled_features(pooled_features)
        
        result = {}
        
        if return_features:
            result["chemical_features"] = chemical_features
            
        if return_reconstruction:
            # Generate reconstruction logits
            reconstruction_logits = self.reconstruction_head(chemical_features)
            result["reconstruction_logits"] = reconstruction_logits
            
        return result

    def extract_pooled_features(self, smiles_list: List[str], batch_size: int = 64) -> torch.Tensor:
        """Return frozen DistilBERT pooled hidden states before the trainable projection."""
        pooled_batches = []
        device = next(self.distilbert_model.parameters()).device
        for start in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[start:start + batch_size]
            batch_encoding = self.tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
            with torch.no_grad():
                outputs = self.distilbert_model(**batch_encoding)
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
        """Project cached DistilBERT pooled hidden states to the metabolite feature space."""
        device = next(self.feature_projection.parameters()).device
        return self.feature_projection(pooled_features.to(device))
    
    def get_smiles_embeddings(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Get chemical embeddings for a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Chemical embeddings tensor [batch_size, hidden_size]
        """
        with torch.no_grad():
            result = self.forward(smiles_list, return_features=True, return_reconstruction=False)
            return result["chemical_features"]


class MetaboliteLearningModule(nn.Module):
    """
    Complete metabolite learning module with training capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "DeepChem/ChemBERTa-77M-MLM",
        hidden_size: int = 256,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        reconstruction_weight: float = 1.0
    ):
        super().__init__()
        
        use_chemberta = model_name and ('chemberta' in model_name.lower() or 'ChemBERTa' in model_name)
        self.encoder = get_metabolite_encoder(
            use_chemberta=bool(use_chemberta),
            model_name=model_name,
            hidden_size=hidden_size,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        self.reconstruction_weight = reconstruction_weight
        
    def forward(
        self,
        smiles_list: List[str],
        targets: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional reconstruction loss.
        
        Args:
            smiles_list: Input SMILES strings
            targets: Target SMILES for reconstruction (optional)
            
        Returns:
            Dictionary with features, loss, and metrics
        """
        
        # Get encoder outputs
        encoder_outputs = self.encoder(
            smiles_list,
            return_features=True,
            return_reconstruction=targets is not None
        )
        
        result = {
            "chemical_features": encoder_outputs["chemical_features"]
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
        Compute reconstruction loss for SMILES prediction.
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            targets: Target SMILES strings
            tokenizer: MolBERT tokenizer
            
        Returns:
            Reconstruction loss
        """
        # Tokenize targets
        target_encoding = tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=256,
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
    
    def get_features(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Extract chemical features for downstream tasks.
        
        Args:
            smiles_list: SMILES strings
            
        Returns:
            Chemical feature embeddings
        """
        return self.encoder.get_smiles_embeddings(smiles_list)


class SMILESSimilarityCalculator:
    """
    Utility class for computing SMILES similarity features.
    """
    
    def __init__(self, fingerprint_type: str = "morgan"):
        self.fingerprint_type = fingerprint_type
        
    def compute_similarity_matrix(self, smiles_list: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Similarity matrix [n_compounds, n_compounds]
        """
        chem, _ = _load_rdkit()
        n_compounds = len(smiles_list)
        similarity_matrix = np.zeros((n_compounds, n_compounds))
        
        for i, smiles1 in enumerate(smiles_list):
            mol1 = chem.MolFromSmiles(smiles1)
            if mol1 is None:
                continue
                
            fp1 = self._get_fingerprint(mol1)
            
            for j, smiles2 in enumerate(smiles_list):
                mol2 = chem.MolFromSmiles(smiles2)
                if mol2 is None:
                    continue
                    
                fp2 = self._get_fingerprint(mol2)
                similarity = self._compute_fingerprint_similarity(fp1, fp2)
                similarity_matrix[i, j] = similarity
                
        return similarity_matrix
    
    def _get_fingerprint(self, mol):
        """Get molecular fingerprint."""
        _, all_chem = _load_rdkit()
        if self.fingerprint_type == "morgan":
            return all_chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        else:
            return all_chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    def _compute_fingerprint_similarity(self, fp1, fp2):
        """Compute Tanimoto similarity between fingerprints."""
        chem, _ = _load_rdkit()
        return chem.DataStructs.TanimotoSimilarity(fp1, fp2)


def load_pretrained_metabolite_model(model_path: str) -> MetaboliteLearningModule:
    """
    Load a pretrained metabolite learning module.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded metabolite learning module
    """
    model = MetaboliteLearningModule()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model


def save_metabolite_model(model: MetaboliteLearningModule, model_path: str):
    """
    Save a trained metabolite learning module.
    
    Args:
        model: Metabolite learning module to save
        model_path: Path to save the model
    """
    torch.save(model.state_dict(), model_path)
