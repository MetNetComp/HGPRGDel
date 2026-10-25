"""GraphGDel entry point (under ``baseline/``). Paths resolve from the repository root.

Inference loads only from ``baseline/GraphGdel_<CBM>.sav``; missing or failed loads use random init.
With ``--train 1``, saves a checkpoint to ``checkpoints/new_code_model_<CBM>.sav``.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend to avoid opening cmd.exe windows in WSL2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import csv
import random
import time
import networkx as nx
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, negative_sampling

from utils.read_gene_sequences import read_gene_sequences
from utils.read_metabolite_smiles import read_metabolite_smiles
from utils.read_gene_metabolite_relationships import read_gene_metabolite_relationships
from utils.create_gene_metabolite_dataset import create_gene_metabolite_dataset
from utils.build_metabolite_graph import load_graph_data as load_graph_data_from_builder
from utils.paths import project_root
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.prediction_threshold import (
    BINARY_THRESHOLD_TRAIN_MACRO_F1,
    BINARY_THRESHOLD_YOUDEN,
    DEFAULT_BINARY_THRESHOLD_MODE,
    select_global_threshold_macro_f1,
    youden_threshold,
)
from baseline.graphgdel_model.GeneLSTM import GeneLSTM
from baseline.graphgdel_model.MetaLSTM import MetaLSTM
from baseline.graphgdel_model.GCN import GCN
from baseline.graphgdel_model.CombinedModel import CombinedModel

###################################################


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Gene Deletion Strategy Prediction with GraphGdel")
    parser.add_argument(
        "--CBM",
        type=str,
        default="e_coli_core",
        help="The CBM model to use (e.g., iML1515, iMM904, e_coli_core)",
    )
    parser.add_argument(
        "--use_cpu",
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to force CPU, 0 to use CUDA if available",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Legacy GraphGdel split seed (1001..1050); ignored when --cv_protocol manuscript",
    )
    parser.add_argument(
        "--cv_protocol",
        choices=["legacy", "manuscript"],
        default="legacy",
        help="manuscript: metabolite-held-out five-fold (needs cv_partitions.json)",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold index 0..4 for manuscript CV")
    parser.add_argument(
        "--output_metrics_csv",
        type=str,
        default=None,
        help="If set, append one row (CBM, method, split_seed, accuracy, macro_precision, macro_recall, macro_f1, macro_auc) to this CSV",
    )
    parser.add_argument(
        "--train",
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to run pre-training and save checkpoint to checkpoints/ (architecture matches current graph)",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=50,
        help="Number of epochs for pre-training when --train 1",
    )

    parser.add_argument(
        "--binary_threshold",
        choices=[BINARY_THRESHOLD_TRAIN_MACRO_F1, BINARY_THRESHOLD_YOUDEN],
        default=DEFAULT_BINARY_THRESHOLD_MODE,
        help="train_macro_f1: global threshold tuned on training metabolites; youden_per_metabolite: legacy per-metabolite ROC",
    )

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    CBM = args.CBM
    use_cpu = args.use_cpu
    print(f"Using CBM: {CBM}")

    # Set the device
    if use_cpu == 1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Set the bound for fixed_gene exclusion
    bound = 0.9
    # Data partitioning
    N_FOLDS = 10
    FIRST_SPLIT_SEED = 1001

    # Hyperparameters
    vocab_embedding_dim = 32  # The size of the embedding vector for each token word
    hidden_dim = 64  # The number of units in the LSTM and GCN hidden layer as output
    gcn_hidden_channels = 64
    batch_size = 8
    epochs = 1
    train_epochs = args.train_epochs

    # Set the random seed for reproducibility
    random_seed = 1001
    random.seed(random_seed)  # For Python's built-in random module
    np.random.seed(random_seed)  # For numpy-based randomness

    repo_root = str(project_root())

    gene_folder_path = os.path.join(repo_root, "Data", CBM, "Gene_AA")
    metabolite_smiles_path = os.path.join(repo_root, "Data", CBM, "Target metabolites_smile.csv")
    relationship_folder_path = os.path.join(repo_root, "Data", CBM, "Label")
    graph_path = os.path.join(repo_root, "Data", CBM, "SMILES_node_feature_final.csv")
    output_csv_path = os.path.join(repo_root, "Data", CBM, "Results", "all_metabolites_predictions_temp.csv")
    ckpt_root = os.path.join(repo_root, "checkpoints")
    baseline_graphgdel_ckpt = os.path.join(repo_root, "baseline", f"GraphGdel_{CBM}.sav")
    new_code_ckpt = os.path.join(ckpt_root, f"new_code_model_{CBM}.sav")

    ###################################################

    # Reading data (exactly like source code)
    print("Loading data...")
    gene_names, gene_sequences = read_gene_sequences(gene_folder_path)
    metabolite_names, smiles_features = read_metabolite_smiles(metabolite_smiles_path)
    relationships = read_gene_metabolite_relationships(relationship_folder_path)

    # Tokenize data exactly like source code
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Tokenizing gene sequences (exactly like source code)
    gene_tokenizer = Tokenizer(char_level=True)
    gene_tokenizer.fit_on_texts(gene_sequences)
    gene_vocab_size = len(gene_tokenizer.word_index) + 1
    max_len = max(len(seq) for seq in gene_sequences)
    gene_sequences = pad_sequences(gene_tokenizer.texts_to_sequences(gene_sequences), maxlen=max_len)

    # Tokenizing SMILES sequences (exactly like source code)
    smiles_tokenizer = Tokenizer(char_level=True)
    smiles_tokenizer.fit_on_texts(smiles_features)
    smiles_vocab_size = len(smiles_tokenizer.word_index) + 1
    smiles_max_len = max(len(smiles) for smiles in smiles_features)
    smiles_features = pad_sequences(smiles_tokenizer.texts_to_sequences(smiles_features), maxlen=smiles_max_len)

    # Extract unique metabolites (needed for graph builder when JSON is used)
    unique_metabolites = list(relationships.keys())
    data_dir = os.path.join(repo_root, "Data")
    graph_data, num_nodes, metabolite_names_order = load_graph_data_from_builder(
        graph_path,
        CBM,
        data_dir=data_dir,
        unique_metabolites=unique_metabolites,
        top_k=10,
        nodes_to_restore=["pyr_c"],
    )
    graph_data = graph_data.to(device)
    if metabolite_names_order is not None:
        graph_node_index_map = {name: i for i, name in enumerate(metabolite_names_order)}
        print("Graph loaded from CBM JSON (manuscript pipeline).")
    else:
        graph_node_index_map = None
        print("Graph: fallback random (no CBM JSON found).")

    train_val_pool = list(unique_metabolites)
    metabolite_labels = np.array([1 if np.mean(relationships[meta][1]) >= 0.5 else 0 for meta in train_val_pool])

    if args.cv_protocol == "manuscript":
        from utils.manuscript_cv import train_val_metabolites

        train_metabolites, val_metabolites = train_val_metabolites(CBM, args.fold, Path(repo_root))
        args.split_seed = 42 + args.fold
    else:
        split_seed = getattr(args, "split_seed", None)
        if split_seed is None:
            split_seed = FIRST_SPLIT_SEED
            args.split_seed = split_seed
        repeat_idx = (split_seed - FIRST_SPLIT_SEED) // N_FOLDS
        fold_idx = (split_seed - FIRST_SPLIT_SEED) % N_FOLDS

        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat_idx)
        X_idx = np.arange(len(train_val_pool))
        folds = list(kf.split(X_idx, metabolite_labels))
        train_idx, val_idx = folds[fold_idx]
        train_metabolites = [train_val_pool[i] for i in train_idx]
        val_metabolites = [train_val_pool[i] for i in val_idx]

    evaluate_fold_holdout = args.output_metrics_csv is not None
    if evaluate_fold_holdout:
        print("Metabolites used for training (this fold):")
        print(train_metabolites[:5], "..." if len(train_metabolites) > 5 else "")
        print("\nMetabolites used for validation (this fold):")
        print(val_metabolites[:5], "..." if len(val_metabolites) > 5 else "")
    else:
        print(
            f"Quick-run evaluation on all {len(unique_metabolites)} target metabolites "
            "(fold split not used for metrics)."
        )

    # Pre-determined genes (>=90% label consistency); excluded from evaluation only.
    gene_count_all: dict = {}
    gene_total_all: dict = {}
    for meta_name in unique_metabolites:
        rel_gene_names, labels = relationships[meta_name]
        for gene_name, label in zip(rel_gene_names, labels):
            if gene_name not in gene_count_all:
                gene_count_all[gene_name] = {"0": 0, "1": 0}
                gene_total_all[gene_name] = 0
            gene_count_all[gene_name][str(label)] += 1
            gene_total_all[gene_name] += 1

    genes_to_exclude: set = set()
    for gene in gene_count_all:
        total = gene_total_all[gene]
        if total <= 0:
            continue
        pct_del = gene_count_all[gene]["0"] / total
        pct_keep = gene_count_all[gene]["1"] / total
        if pct_del >= bound or pct_keep >= bound:
            genes_to_exclude.add(gene)
    print(f"Genes to exclude during evaluation: {len(genes_to_exclude)}")

    # Create datasets
    print("Creating datasets...")

    # Custom collate function for PyTorch Geometric Data objects
    def custom_collate(batch):
        gene_seqs = torch.stack([item[0] for item in batch])
        smiles_seqs = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        node_indices = torch.stack([item[3] for item in batch])
        # For graph data, we'll use the first one since they're all the same
        graph = batch[0][4]
        return gene_seqs, smiles_seqs, labels, node_indices, graph

    # Vocabulary sizes already calculated above (exactly like source code)

    # Create datasets (data already tokenized like source code)
    train_dataset = create_gene_metabolite_dataset(
        train_metabolites,
        relationships,
        gene_sequences,
        smiles_features,
        list(gene_names),
        graph_data,
        device,
        gene_names=gene_names,
        metabolite_names=metabolite_names,
        graph_node_index_map=graph_node_index_map,
    )

    val_dataset = create_gene_metabolite_dataset(
        val_metabolites,
        relationships,
        gene_sequences,
        smiles_features,
        list(gene_names),
        graph_data,
        device,
        gene_names=gene_names,
        metabolite_names=metabolite_names,
        graph_node_index_map=graph_node_index_map,
    )

    # Custom collate function for PyTorch Geometric Data objects
    def custom_collate(batch):
        gene_seqs = torch.stack([item[0] for item in batch])
        smiles_seqs = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])
        node_indices = torch.stack([item[3] for item in batch])
        # For graph data, we'll use the first one since they're all the same
        graph = batch[0][4]
        return gene_seqs, smiles_seqs, labels, node_indices, graph

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    # Initialize models
    print("Initializing models...")

    gene_lstm = GeneLSTM(vocab_size=gene_vocab_size, vocab_embedding_dim=vocab_embedding_dim, hidden_dim=hidden_dim)

    smiles_lstm = MetaLSTM(vocab_size=smiles_vocab_size, vocab_embedding_dim=vocab_embedding_dim, hidden_dim=hidden_dim)

    gcn_model = GCN(
        in_channels=graph_data.x.size(-1),  # Use actual graph feature dimension
        hidden_channels=gcn_hidden_channels,
        out_channels=hidden_dim,
        num_nodes=num_nodes,
    )

    model = CombinedModel(gene_lstm, smiles_lstm, gcn_model, hidden_dim)
    model = model.to(device)

    # Load pretrained model (or run pre-training to produce a matching checkpoint)
    if args.train:
        print(f"Pre-training for {train_epochs} epochs and saving checkpoint to {new_code_ckpt}")
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(train_epochs):
            running_loss = 0.0
            for batch in train_loader:
                gene_seqs, smiles_seqs, labels, node_indices, graph = batch
                gene_seqs = gene_seqs.to(device)
                smiles_seqs = smiles_seqs.to(device)
                labels = labels.to(device).unsqueeze(1)
                node_indices = node_indices.to(device)
                graph = graph.to(device)
                optimizer.zero_grad()
                out, _, _ = model(gene_seqs, smiles_seqs, graph, node_indices)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{train_epochs} loss: {running_loss/len(train_loader):.4f}")
        os.makedirs(ckpt_root, exist_ok=True)
        torch.save(model.state_dict(), new_code_ckpt)
        print(f"Saved checkpoint to {new_code_ckpt} (matches current graph: {num_nodes} nodes).")
    else:
        if os.path.isfile(baseline_graphgdel_ckpt):
            try:
                checkpoint = torch.load(
                    baseline_graphgdel_ckpt, map_location=device, weights_only=True
                )
                model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded pretrained model {baseline_graphgdel_ckpt}")
            except Exception as e:
                print(
                    f"Warning: Could not load {baseline_graphgdel_ckpt} ({e}). "
                    "Running with randomly initialized weights."
                )
        else:
            print(
                f"No checkpoint at {baseline_graphgdel_ckpt}; running with randomly initialized weights.\n"
                "Place baseline/GraphGdel_<CBM>.sav, or use --train 1 to save checkpoints/new_code_model_<CBM>.sav."
            )

    # Evaluation using the same method as source code
    print("Evaluating model...")
    model.eval()

    def evaluate_per_metabolite(
        model,
        metabolite_names,
        gene_names,
        gene_sequences,
        smiles_features,
        relationships,
        graph,
        device,
        genes_to_exclude=set(),
        gene_name_to_index=None,
        val_metabolites=None,
        graph_node_index_map=None,
        train_metabolites=None,
        binary_threshold=DEFAULT_BINARY_THRESHOLD_MODE,
    ):
        model.eval()
        graph = graph.to(device)

        if graph_node_index_map is not None:
            metabolite_to_node_idx = graph_node_index_map
        else:
            metabolite_to_node_idx = {met: idx for idx, met in enumerate(metabolite_names)}

        gene_seq_dict = {
            gene: torch.tensor(gene_sequences[gene_name_to_index[gene]], dtype=torch.float32, device=device).unsqueeze(0)
            for gene in gene_names
            if gene in gene_name_to_index
        }

        smiles_dict = {
            met: torch.tensor(smiles_features[idx], dtype=torch.float32, device=device).unsqueeze(0)
            for idx, met in enumerate(metabolite_names)
        }

        def predict_for_metabolite(metabolite):
            if metabolite not in relationships or metabolite not in metabolite_to_node_idx:
                return None
            gene_list, label_list = relationships[metabolite]
            filtered = [(g, l) for g, l in zip(gene_list, label_list) if g not in genes_to_exclude and g in gene_seq_dict]
            if len(filtered) == 0:
                return None
            genes_filtered, labels_filtered = zip(*filtered)
            labels_np = np.array(labels_filtered)
            node_idx = metabolite_to_node_idx[metabolite]
            node_idx_tensor = torch.tensor([node_idx], dtype=torch.long, device=device)
            preds_probs = []
            for gene in genes_filtered:
                gene_seq = gene_seq_dict[gene]
                smiles_seq = smiles_dict[metabolite]
                with torch.no_grad():
                    output, _, _ = model(gene_seq, smiles_seq, graph, node_idx_tensor)
                    pred_prob = output.squeeze().cpu().item()
                    preds_probs.append(pred_prob)
            return {
                "true": labels_np,
                "pred_probs": np.array(preds_probs),
                "genes": genes_filtered,
            }

        global_threshold = None
        if binary_threshold == BINARY_THRESHOLD_TRAIN_MACRO_F1:
            if not train_metabolites:
                raise ValueError("train_metabolites required when binary_threshold=train_macro_f1")
            tune_labels: list[int] = []
            tune_scores: list[float] = []
            for metabolite in train_metabolites:
                info = predict_for_metabolite(metabolite)
                if info is None:
                    continue
                tune_labels.extend(int(v) for v in info["true"])
                tune_scores.extend(float(v) for v in info["pred_probs"])
            global_threshold = select_global_threshold_macro_f1(tune_labels, tune_scores)
            print(f"Global threshold from training metabolites (macro-F1): {global_threshold:.4f}")

        per_metabolite_info = {}
        metabolites_to_score = val_metabolites if val_metabolites is not None else metabolite_names
        for metabolite in metabolites_to_score:
            info = predict_for_metabolite(metabolite)
            if info is None:
                continue
            labels_np = info["true"]
            preds_probs_np = info["pred_probs"]

            if binary_threshold == BINARY_THRESHOLD_TRAIN_MACRO_F1:
                best_thresh = global_threshold
            else:
                best_thresh = youden_threshold(labels_np, preds_probs_np)

            pred_labels = (preds_probs_np >= best_thresh).astype(int)
            best_f1 = f1_score(labels_np, pred_labels, average="macro", zero_division=0)

            per_metabolite_info[metabolite] = {
                "true": labels_np,
                "pred_probs": preds_probs_np,
                "pred_labels": pred_labels,
                "genes": info["genes"],
                "best_threshold": best_thresh,
                "best_f1": best_f1,
            }

            print(f"Metabolite: {metabolite}")
            print(f"  Number of Non-fixed Genes: {len(labels_np)}")
            print(f"  Threshold: {best_thresh:.3f} with F1: {best_f1:.3f}")
            print(f"  Accuracy: {accuracy_score(labels_np, pred_labels)*100:.2f}%")
            print(f"  Precision: {precision_score(labels_np, pred_labels, average='macro', zero_division=0)*100:.2f}%")
            print(f"  Recall: {recall_score(labels_np, pred_labels, average='macro', zero_division=0)*100:.2f}%")
            print(f"  F1 Score: {best_f1*100:.2f}%")
            print(f"  True labels:      {np.array(labels_np[:10])}")
            print(f"  Pred probabilities: {np.round(preds_probs_np[:10], 2)}")
            print(f"  Pred labels:     {np.array(pred_labels[:10])}\n")

        acc_list = []
        prec_list = []
        rec_list = []
        f1_list = []

        for metabolite, info in per_metabolite_info.items():
            labels_np = info["true"]
            pred_labels = info["pred_labels"]
            acc_list.append(accuracy_score(labels_np, pred_labels) * 100)
            prec_list.append(precision_score(labels_np, pred_labels, average="macro", zero_division=0) * 100)
            rec_list.append(recall_score(labels_np, pred_labels, average="macro", zero_division=0) * 100)
            f1_list.append(f1_score(labels_np, pred_labels, average="macro", zero_division=0) * 100)

        print("====================== GraphGDel baseline Report ======================")
        mean_acc = np.mean(acc_list) if acc_list else 0.0
        mean_prec = np.mean(prec_list) if prec_list else 0.0
        mean_rec = np.mean(rec_list) if rec_list else 0.0
        mean_f1 = np.mean(f1_list) if f1_list else 0.0
        print(f"Overall Accuracy: {mean_acc:.2f}%")
        print(f"Macro-Averaged Precision: {mean_prec:.2f}%")
        print(f"Macro-Averaged Recall: {mean_rec:.2f}%")
        print(f"Macro-Averaged F1 Score: {mean_f1:.2f}%")
        all_true, all_probs = [], []
        for metabolite, info in per_metabolite_info.items():
            all_true.extend(info["true"].tolist())
            all_probs.extend(info["pred_probs"].tolist())
        try:
            from sklearn.metrics import roc_auc_score

            mean_auc = roc_auc_score(all_true, all_probs) * 100 if all_true else 0.0
        except Exception:
            mean_auc = 0.0
        return (mean_acc, mean_prec, mean_rec, mean_f1, mean_auc)

    val_for_csv = val_metabolites if getattr(args, "output_metrics_csv", None) else None
    eval_metabolites = val_for_csv if val_for_csv is not None else unique_metabolites
    summary = evaluate_per_metabolite(
        model=model,
        metabolite_names=metabolite_names,
        gene_names=gene_names,
        gene_sequences=gene_sequences,
        smiles_features=smiles_features,
        relationships=relationships,
        graph=graph_data,
        device=device,
        genes_to_exclude=set(genes_to_exclude),
        gene_name_to_index={gene: idx for idx, gene in enumerate(gene_names)},
        val_metabolites=eval_metabolites,
        graph_node_index_map=graph_node_index_map,
        train_metabolites=train_metabolites,
        binary_threshold=args.binary_threshold,
    )
    if args.output_metrics_csv and summary is not None:
        import csv

        write_header = not os.path.exists(args.output_metrics_csv)
        with open(args.output_metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "CBM",
                        "method",
                        "split_seed",
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "macro_auc",
                    ]
                )
            w.writerow(
                [
                    CBM,
                    "GraphGdel",
                    args.split_seed,
                    round(summary[0], 4),
                    round(summary[1], 4),
                    round(summary[2], 4),
                    round(summary[3], 4),
                    round(summary[4], 4),
                ]
            )

    # Save results
    print(f"Saving results to {output_csv_path}")
    # Implementation for saving results would go here

    print("GraphGdel training and evaluation completed!")


if __name__ == "__main__":
    main()
