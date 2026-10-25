#!/usr/bin/env python3
"""
Training and Evaluation Module for HGPRGDel
===========================================

This module contains the training and evaluation logic for the HGPRGDel model,
including cross-validation, metrics calculation, and model selection.

Key Components:
- train_epoch: Training function for a single epoch
- validate_epoch: Validation function for a single epoch
- evaluate_model: Model evaluation with comprehensive metrics
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional


def train_epoch(model, dataloader, criterion, optimizer, device, reaction_compounds, reaction_genes, 
                compound_reaction_edges, reaction_gene_edges, gpr_rules, gene_to_idx, evaluation_genes=None,
                reconstruction_weight=0.0):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        reaction_compounds: Reaction-compound mappings
        reaction_genes: Reaction-gene mappings
        compound_reaction_edges: Compound-reaction edges
        reaction_gene_edges: Reaction-gene edges
        gpr_rules: GPR rules for reactions
        gene_to_idx: Gene name to index mapping
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        gene_sequences = batch['gene_sequence']
        metabolite_smiles = batch['metabolite_smiles']
        labels = batch['label'].float().to(device)
        
        optimizer.zero_grad()
        
        model_output = model(gene_sequences, metabolite_smiles, reaction_compounds, reaction_genes,
                           compound_reaction_edges, reaction_gene_edges, gpr_rules, gene_to_idx,
                           return_reconstruction_loss=False)
        
        if isinstance(model_output, tuple):
            predictions, _ = model_output
        else:
            predictions = model_output
        
        # Calculate main classification loss
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if predictions.dim() == 0:  # If scalar, add batch dimension
            predictions = predictions.unsqueeze(0)
        total_batch_loss = criterion(predictions, labels)
        
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        
        # Store predictions and labels for metrics
        # Only include evaluation genes in metrics if specified
        if evaluation_genes is None:
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        else:
            # Filter to only include evaluation genes
            for i, (pred, label) in enumerate(zip(predictions.detach().cpu().numpy(), labels.detach().cpu().numpy())):
                gene_name = batch['gene_name'][i]
                if gene_name in evaluation_genes:
                    all_predictions.append(pred)
                    all_labels.append(label)
    
    avg_loss = total_loss / len(dataloader)
    
    if len(all_predictions) > 0:
        accuracy = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_predictions])
    else:
        accuracy = 0.0
    
    return avg_loss, accuracy, 0.0, 0.0


def validate_epoch_with_metabolite_thresholds(model, dataloader, criterion, device, reaction_compounds, reaction_genes,
                                             compound_reaction_edges, reaction_gene_edges, gpr_rules, gene_to_idx, 
                                             evaluation_genes=None, reconstruction_weight=0.0):
    """
    Validate the model with metabolite-specific threshold selection.
    
    This function:
    1. Collects predictions per metabolite
    2. Finds optimal threshold for each metabolite using ROC curve
    3. Calculates metrics using metabolite-specific thresholds
    
    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        reaction_compounds: Reaction-compound mappings
        reaction_genes: Reaction-gene mappings
        compound_reaction_edges: Compound-reaction edges
        reaction_gene_edges: Reaction-gene edges
        gpr_rules: GPR rules for reactions
        gene_to_idx: Gene name to index mapping
        evaluation_genes: Genes to include in evaluation (optional)
        reconstruction_weight: Unused (reserved for API compatibility).
        
    Returns:
        Tuple of (overall_metrics, metabolite_metrics, metabolite_thresholds)
    """
    model.eval()
    total_loss = 0.0
    
    # Collect predictions per metabolite
    metabolite_data = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            gene_sequences = batch['gene_sequence']
            metabolite_smiles = batch['metabolite_smiles']
            labels = batch['label'].float().to(device)
            
            model_output = model(gene_sequences, metabolite_smiles, reaction_compounds, reaction_genes,
                               compound_reaction_edges, reaction_gene_edges, gpr_rules, gene_to_idx,
                               return_reconstruction_loss=False)
            
            if isinstance(model_output, tuple):
                predictions, _ = model_output
            else:
                predictions = model_output
            
            # Calculate main classification loss
            if predictions.dim() > 1:
                predictions = predictions.squeeze()
            if predictions.dim() == 0:  # If scalar, add batch dimension
                predictions = predictions.unsqueeze(0)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            # Collect predictions per metabolite
            for i, (pred, metabolite, label, gene_name) in enumerate(zip(
                predictions.detach().cpu().numpy(), 
                batch['metabolite_name'], 
                labels.detach().cpu().numpy(),
                batch['gene_name']
            )):
                # Only include evaluation genes if specified
                if evaluation_genes is None or gene_name in evaluation_genes:
                    if metabolite not in metabolite_data:
                        metabolite_data[metabolite] = {'predictions': [], 'labels': []}
                    
                    metabolite_data[metabolite]['predictions'].append(pred)
                    metabolite_data[metabolite]['labels'].append(label)
    
    # Find optimal threshold for each metabolite
    metabolite_thresholds = {}
    metabolite_metrics = {}
    
    for metabolite, data in metabolite_data.items():
        preds = data['predictions']
        labels = data['labels']
        
        if len(preds) >= 10:  # Minimum samples for threshold selection
            try:
                # Find optimal threshold using Youden's J statistic (maximize TPR - FPR)
                fpr, tpr, thresholds = roc_curve(labels, preds)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold = thresholds[optimal_idx]
                metabolite_thresholds[metabolite] = optimal_threshold
                
                # Calculate metrics with optimal threshold
                binary_preds = [1 if p > optimal_threshold else 0 for p in preds]
                metabolite_metrics[metabolite] = {
                    'threshold': optimal_threshold,
                    'accuracy': accuracy_score(labels, binary_preds),
                    'f1': f1_score(labels, binary_preds, zero_division=0),
                    'precision': precision_score(labels, binary_preds, zero_division=0),
                    'recall': recall_score(labels, binary_preds, zero_division=0),
                    'samples': len(preds),
                    'keep_ratio': sum(labels) / len(labels)
                }
            except ValueError:
                # Fallback to default threshold if ROC curve calculation fails
                metabolite_thresholds[metabolite] = 0.5
                binary_preds = [1 if p > 0.5 else 0 for p in preds]
                metabolite_metrics[metabolite] = {
                    'threshold': 0.5,
                    'accuracy': accuracy_score(labels, binary_preds),
                    'f1': f1_score(labels, binary_preds, zero_division=0),
                    'precision': precision_score(labels, binary_preds, zero_division=0),
                    'recall': recall_score(labels, binary_preds, zero_division=0),
                    'samples': len(preds),
                    'keep_ratio': sum(labels) / len(labels)
                }
        else:
            # Use default threshold for small samples
            metabolite_thresholds[metabolite] = 0.5
            binary_preds = [1 if p > 0.5 else 0 for p in preds]
            metabolite_metrics[metabolite] = {
                'threshold': 0.5,
                'accuracy': accuracy_score(labels, binary_preds),
                'f1': f1_score(labels, binary_preds, zero_division=0),
                'precision': precision_score(labels, binary_preds, zero_division=0),
                'recall': recall_score(labels, binary_preds, zero_division=0),
                'samples': len(preds),
                'keep_ratio': sum(labels) / len(labels)
            }
    
    # Calculate overall metrics using metabolite-specific thresholds
    all_predictions = []
    all_labels = []
    all_binary_preds = []
    
    for metabolite, data in metabolite_data.items():
        preds = data['predictions']
        labels = data['labels']
        threshold = metabolite_thresholds[metabolite]
        binary_preds = [1 if p > threshold else 0 for p in preds]
        
        all_predictions.extend(preds)
        all_labels.extend(labels)
        all_binary_preds.extend(binary_preds)
    
    # Calculate overall metrics
    avg_loss = total_loss / len(dataloader)
    
    if len(all_predictions) > 0:
        overall_metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_binary_preds),
            'f1': f1_score(all_labels, all_binary_preds, zero_division=0),
            'precision': precision_score(all_labels, all_binary_preds, zero_division=0),
            'recall': recall_score(all_labels, all_binary_preds, zero_division=0)
        }
        
        # Calculate AUC using raw predictions
        try:
            overall_metrics['auc'] = roc_auc_score(all_labels, all_predictions)
        except ValueError:
            overall_metrics['auc'] = 0.5
    else:
        overall_metrics = {
            'loss': avg_loss,
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'auc': 0.5
        }
    
    return overall_metrics, metabolite_metrics, metabolite_thresholds


def validate_epoch(model, dataloader, criterion, device, reaction_compounds, reaction_genes,
                   compound_reaction_edges, reaction_gene_edges, gpr_rules, gene_to_idx, evaluation_genes=None,
                   reconstruction_weight=0.0):
    """Validate the model for one epoch using a fixed 0.5 classification threshold."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            gene_sequences = batch['gene_sequence']
            metabolite_smiles = batch['metabolite_smiles']
            labels = batch['label'].float().to(device)
            
            model_output = model(gene_sequences, metabolite_smiles, reaction_compounds, reaction_genes,
                               compound_reaction_edges, reaction_gene_edges, gpr_rules, gene_to_idx,
                               return_reconstruction_loss=False)
            
            if isinstance(model_output, tuple):
                predictions, _ = model_output
            else:
                predictions = model_output
            
            # Calculate main classification loss
            if predictions.dim() > 1:
                predictions = predictions.squeeze()
            if predictions.dim() == 0:  # If scalar, add batch dimension
                predictions = predictions.unsqueeze(0)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            # Store predictions and labels for metrics
            # Only include evaluation genes in metrics if specified
            if evaluation_genes is None:
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
            else:
                # Filter to only include evaluation genes
                for i, (pred, label) in enumerate(zip(predictions.detach().cpu().numpy(), labels.detach().cpu().numpy())):
                    gene_name = batch['gene_name'][i]
                    if gene_name in evaluation_genes:
                        all_predictions.append(pred)
                        all_labels.append(label)
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    if len(all_predictions) > 0:
        binary_predictions = [1 if p > 0.5 else 0 for p in all_predictions]
        accuracy = accuracy_score(all_labels, binary_predictions)
        f1 = f1_score(all_labels, binary_predictions, average="macro", zero_division=0)
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, all_predictions)
        except ValueError:
            auc = 0.5  # Default AUC if only one class present
    else:
        accuracy = 0.0
        f1 = 0.0
        auc = 0.5
    
    return avg_loss, accuracy, f1, auc, 0.0, 0.0


def evaluate_model(model, dataloader, device, reaction_compounds, reaction_genes,
                   compound_reaction_edges, reaction_gene_edges, gpr_rules, gene_to_idx):
    """
    Evaluate model with comprehensive metrics.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader
        device: Device to run on
        reaction_compounds: Reaction-compound mappings
        reaction_genes: Reaction-gene mappings
        compound_reaction_edges: Compound-reaction edges
        reaction_gene_edges: Reaction-gene edges
        gpr_rules: GPR rules for reactions
        gene_to_idx: Gene name to index mapping
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            gene_sequences = batch['gene_sequence']
            metabolite_smiles = batch['metabolite_smiles']
            labels = batch['label'].float().to(device)
            
            predictions = model(gene_sequences, metabolite_smiles, reaction_compounds, reaction_genes,
                               compound_reaction_edges, reaction_gene_edges, gpr_rules, gene_to_idx)
            
            if predictions.dim() > 1:
                predictions = predictions.squeeze()
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    
    # Calculate metrics
    binary_predictions = [1 if p > 0.5 else 0 for p in all_predictions]
    
    metrics = {
        'accuracy': accuracy_score(all_labels, binary_predictions),
        'precision': precision_score(all_labels, binary_predictions, zero_division=0),
        'recall': recall_score(all_labels, binary_predictions, zero_division=0),
        'f1_score': f1_score(all_labels, binary_predictions, zero_division=0)
    }
    
    # Calculate AUC
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        metrics['auc'] = 0.5
    
    return metrics
