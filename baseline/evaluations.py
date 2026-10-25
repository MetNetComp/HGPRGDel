import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid opening cmd.exe windows in WSL2
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import time
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from utils.prediction_threshold import (
    BINARY_THRESHOLD_TRAIN_MACRO_F1,
    BINARY_THRESHOLD_YOUDEN,
    DEFAULT_BINARY_THRESHOLD_MODE,
    BinaryThresholdMode,
    select_global_threshold_macro_f1,
    youden_threshold,
)


def predict_gene_deletions_for_metabolite(model, metabolite_name, metabolite_names, gene_sequences, smiles_features, relationships, device, genes_to_exclude):
    # Find the index of the metabolite
    meta_index = metabolite_names.tolist().index(metabolite_name)
    fingerprint_feature = smiles_features[meta_index]  # shape: (F,)

    model.eval()
    with torch.no_grad():
        fingerprint_tensor = torch.tensor(fingerprint_feature, dtype=torch.float32).to(device)
        gene_seqs_tensor = torch.tensor(gene_sequences, dtype=torch.long).to(device)

        # Repeat fingerprint for each gene
        fingerprint_feat = fingerprint_tensor.expand(len(gene_sequences), -1).contiguous()

        # Model prediction
        output = model(gene_seqs_tensor, fingerprint_feat)
        prediction = output[0] if isinstance(output, tuple) else output
        predicted_probs = torch.sigmoid(prediction).view(-1).cpu().numpy()

    # Extract true labels
    if metabolite_name in relationships:
        gene_names, true_labels = relationships[metabolite_name]
        true_labels = np.array(true_labels)
    else:
        print(f"No relationship data found for metabolite '{metabolite_name}'")
        return [], [], predicted_probs, []

    # Filter out excluded genes
    filtered_indices = [i for i, gene in enumerate(gene_names) if gene not in genes_to_exclude]
    gene_names = [gene_names[i] for i in filtered_indices]
    predicted_probs = predicted_probs[filtered_indices]
    true_labels = true_labels[filtered_indices]

    return gene_names, predicted_probs, true_labels


def _global_threshold_from_metabolites(
    model,
    metabolites,
    metabolite_names,
    gene_sequences,
    smiles_features,
    relationships,
    device,
    genes_to_exclude,
    predict_fn,
    predict_fn_kwargs=None,
):
    labels: list[int] = []
    scores: list[float] = []
    predict_fn_kwargs = predict_fn_kwargs or {}
    for metabolite_name in metabolites:
        gene_names, predicted_probs, true_labels = predict_fn(
            model,
            metabolite_name,
            metabolite_names,
            gene_sequences,
            smiles_features,
            relationships,
            device,
            genes_to_exclude,
            **predict_fn_kwargs,
        )
        if len(gene_names) == 0:
            continue
        labels.extend(int(v) for v in true_labels)
        scores.extend(float(v) for v in predicted_probs)
    return select_global_threshold_macro_f1(labels, scores)


def _macro_metrics_for_metabolite(true_labels, predicted_deletions, predicted_probs):
    num_correct = np.sum(predicted_deletions == true_labels)
    accuracy = (num_correct / len(true_labels)) * 100

    true_positive = [0, 0]
    false_positive = [0, 0]
    false_negative = [0, 0]
    for label in [0, 1]:
        true_positive[label] = np.sum((predicted_deletions == label) & (true_labels == label))
        false_positive[label] = np.sum((predicted_deletions == label) & (true_labels != label))
        false_negative[label] = np.sum((predicted_deletions != label) & (true_labels == label))

    precision, recall, f1_score_vals = [], [], []
    for label in [0, 1]:
        precision_val = (
            true_positive[label] / (true_positive[label] + false_positive[label]) * 100
        ) if (true_positive[label] + false_positive[label]) > 0 else 0.0
        recall_val = (
            true_positive[label] / (true_positive[label] + false_negative[label]) * 100
        ) if (true_positive[label] + false_negative[label]) > 0 else 0.0
        f1_score_val = (
            (2 * precision_val * recall_val) / (precision_val + recall_val)
        ) if (precision_val + recall_val) > 0 else 0.0
        precision.append(precision_val)
        recall.append(recall_val)
        f1_score_vals.append(f1_score_val)

    try:
        auc = roc_auc_score(true_labels, predicted_probs) * 100
    except ValueError:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1_score': float(np.mean(f1_score_vals)),
        'auc': auc,
        'confusion': {
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative,
            'num_correct': int(num_correct),
            'num_genes': len(true_labels),
        },
    }


def calculate_metrics_for_val_metabolites(
    model,
    val_metabolites,
    metabolite_names,
    gene_sequences,
    smiles_features,
    relationships,
    device,
    genes_to_exclude,
    method_name: str = "DNN",
    train_metabolites=None,
    binary_threshold: BinaryThresholdMode = DEFAULT_BINARY_THRESHOLD_MODE,
):
    global_threshold = None
    if binary_threshold == BINARY_THRESHOLD_TRAIN_MACRO_F1:
        if not train_metabolites:
            raise ValueError("train_metabolites required when binary_threshold=train_macro_f1")
        global_threshold = _global_threshold_from_metabolites(
            model,
            train_metabolites,
            metabolite_names,
            gene_sequences,
            smiles_features,
            relationships,
            device,
            genes_to_exclude,
            predict_gene_deletions_for_metabolite,
        )
        print(f"Global threshold from training metabolites (macro-F1): {global_threshold:.4f}")

    total_correct = 0
    total_genes = 0
    total_true_positive = [0, 0]
    total_false_positive = [0, 0]
    total_true_negative = [0, 0]
    total_false_negative = [0, 0]
    metabolite_metrics = []

    for metabolite_name in val_metabolites:
        gene_names, predicted_probs, true_labels = predict_gene_deletions_for_metabolite(
            model, metabolite_name, metabolite_names, gene_sequences, smiles_features, relationships, device, genes_to_exclude
        )

        if len(gene_names) == 0:
            continue

        if binary_threshold == BINARY_THRESHOLD_TRAIN_MACRO_F1:
            predicted_deletions = (predicted_probs >= global_threshold).astype(int)
        else:
            threshold = youden_threshold(true_labels, predicted_probs)
            predicted_deletions = (predicted_probs >= threshold).astype(int)

        metrics = _macro_metrics_for_metabolite(true_labels, predicted_deletions, predicted_probs)
        total_correct += metrics['confusion']['num_correct']
        total_genes += metrics['confusion']['num_genes']

        for label in [0, 1]:
            tp = metrics['confusion']['true_positive'][label]
            fp = metrics['confusion']['false_positive'][label]
            fn = metrics['confusion']['false_negative'][label]
            tn = metrics['confusion']['num_correct'] - tp
            total_true_positive[label] += tp
            total_false_positive[label] += fp
            total_false_negative[label] += fn
            total_true_negative[label] += tn

        metabolite_metrics.append(metrics)

        print(f"Metabolite: {metabolite_name}")
        print(f"  First 10 True Gene Status: {true_labels[:10]}")
        print(f"  First 10 Predicted Gene Status: {predicted_deletions[:10]}")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.2f}% (Macro-Averaged)")
        print(f"  Recall: {metrics['recall']:.2f}% (Macro-Averaged)")
        print(f"  F1 Score: {metrics['f1_score']:.2f}% (Macro-Averaged)")
        print(f"  AUC: {metrics['auc']:.2f}%")
        print()

    overall_accuracy = (total_correct / total_genes) * 100 if total_genes > 0 else 0.0
    average_precision = np.mean([metrics['precision'] for metrics in metabolite_metrics]) if metabolite_metrics else 0.0
    average_recall = np.mean([metrics['recall'] for metrics in metabolite_metrics]) if metabolite_metrics else 0.0
    average_f1_score = np.mean([metrics['f1_score'] for metrics in metabolite_metrics]) if metabolite_metrics else 0.0
    average_auc = np.mean([metrics['auc'] for metrics in metabolite_metrics]) if metabolite_metrics else 0.0

    print()
    print(f"====================== {method_name} baseline Report ======================")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Macro-Averaged Precision: {average_precision:.2f}%")
    print(f"Macro-Averaged Recall: {average_recall:.2f}%")
    print(f"Macro-Averaged F1 Score: {average_f1_score:.2f}%")
    print(f"Macro-Averaged AUC: {average_auc:.2f}%")
    print()
    return (overall_accuracy, average_precision, average_recall, average_f1_score, average_auc)


def predict_gene_deletions_for_metabolite_nma(model, metabolite_name, metabolite_names, metabolite_names_order, gene_sequences, smiles_features, relationships, device, genes_to_exclude, Z_meta_nma):
    """NMA variant: use Z_meta_nma[node_id] for prediction. node_id = index in metabolite_names_order."""
    meta_index = metabolite_names.tolist().index(metabolite_name)
    smiles_row = smiles_features[meta_index]
    try:
        node_id = metabolite_names_order.index(metabolite_name)
    except ValueError:
        node_id = 0

    model.eval()
    with torch.no_grad():
        gene_seqs_tensor = torch.tensor(gene_sequences, dtype=torch.long).to(device)
        smiles_tensor = torch.tensor(smiles_row, dtype=torch.long).to(device).unsqueeze(0).expand(len(gene_sequences), -1)
        meta_indices = torch.full((len(gene_sequences),), node_id, dtype=torch.long, device=device)
        Z_meta_nma_dev = Z_meta_nma.to(device)

        output, _, _ = model(gene_seqs_tensor, smiles_tensor, meta_indices, Z_meta_nma_dev)
        predicted_probs = output.view(-1).cpu().numpy()

    if metabolite_name in relationships:
        gene_names, true_labels = relationships[metabolite_name]
        true_labels = np.array(true_labels)
    else:
        print(f"No relationship data found for metabolite '{metabolite_name}'")
        return [], [], predicted_probs, []

    filtered_indices = [i for i, gene in enumerate(gene_names) if gene not in genes_to_exclude]
    gene_names = [gene_names[i] for i in filtered_indices]
    predicted_probs = predicted_probs[filtered_indices]
    true_labels = true_labels[filtered_indices]
    return gene_names, predicted_probs, true_labels


def calculate_metrics_for_val_metabolites_nma(
    model,
    val_metabolites,
    metabolite_names,
    metabolite_names_order,
    gene_sequences,
    smiles_features,
    relationships,
    device,
    genes_to_exclude,
    Z_meta_nma,
    train_metabolites=None,
    binary_threshold: BinaryThresholdMode = DEFAULT_BINARY_THRESHOLD_MODE,
):
    """Aggregate metrics over val_metabolites using NMA prediction."""
    global_threshold = None
    if binary_threshold == BINARY_THRESHOLD_TRAIN_MACRO_F1:
        if not train_metabolites:
            raise ValueError("train_metabolites required when binary_threshold=train_macro_f1")
        tune_labels: list[int] = []
        tune_scores: list[float] = []
        for metabolite_name in train_metabolites:
            _, predicted_probs, true_labels = predict_gene_deletions_for_metabolite_nma(
                model,
                metabolite_name,
                metabolite_names,
                metabolite_names_order,
                gene_sequences,
                smiles_features,
                relationships,
                device,
                genes_to_exclude,
                Z_meta_nma,
            )
            if len(predicted_probs) == 0:
                continue
            tune_labels.extend(int(v) for v in true_labels)
            tune_scores.extend(float(v) for v in predicted_probs)
        global_threshold = select_global_threshold_macro_f1(tune_labels, tune_scores)
        print(f"Global threshold from training metabolites (macro-F1): {global_threshold:.4f}")

    total_correct = 0
    total_genes = 0
    metabolite_metrics = []

    for metabolite_name in val_metabolites:
        gene_names, predicted_probs, true_labels = predict_gene_deletions_for_metabolite_nma(
            model, metabolite_name, metabolite_names, metabolite_names_order, gene_sequences, smiles_features, relationships, device, genes_to_exclude, Z_meta_nma
        )
        if len(gene_names) == 0:
            continue

        if binary_threshold == BINARY_THRESHOLD_TRAIN_MACRO_F1:
            predicted_deletions = (predicted_probs >= global_threshold).astype(int)
        else:
            threshold = youden_threshold(true_labels, predicted_probs)
            predicted_deletions = (predicted_probs >= threshold).astype(int)

        metrics = _macro_metrics_for_metabolite(true_labels, predicted_deletions, predicted_probs)
        total_correct += metrics['confusion']['num_correct']
        total_genes += metrics['confusion']['num_genes']
        metabolite_metrics.append(metrics)
        print(f"Metabolite: {metabolite_name}")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%  Macro F1: {metrics['f1_score']:.2f}%  AUC: {metrics['auc']:.2f}%")

    overall_accuracy = (total_correct / total_genes) * 100 if total_genes > 0 else 0.0
    average_precision = np.mean([m['precision'] for m in metabolite_metrics]) if metabolite_metrics else 0.0
    average_recall = np.mean([m['recall'] for m in metabolite_metrics]) if metabolite_metrics else 0.0
    average_f1_score = np.mean([m['f1_score'] for m in metabolite_metrics]) if metabolite_metrics else 0.0
    average_auc = np.mean([m['auc'] for m in metabolite_metrics]) if metabolite_metrics else 0.0
    print()
    print("====================== NMA baseline Report ======================")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Macro-Averaged Precision: {average_precision:.2f}%")
    print(f"Macro-Averaged Recall: {average_recall:.2f}%")
    print(f"Macro-Averaged F1 Score: {average_f1_score:.2f}%")
    print(f"Macro-Averaged AUC: {average_auc:.2f}%")
    print()
    return (overall_accuracy, average_precision, average_recall, average_f1_score, average_auc)
