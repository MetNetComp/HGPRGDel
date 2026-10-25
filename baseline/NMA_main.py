"""
NMA (Neighborhood Mean Aggregation) baseline: DeepGDel with prediction using
1-hop neighborhood mean of Meta-M outputs as metabolite feature.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import csv
from pathlib import Path

from utils.read_gene_sequences import read_gene_sequences
from utils.read_metabolite_smiles import read_metabolite_smiles
from utils.read_gene_metabolite_relationships import read_gene_metabolite_relationships
from baseline.create_dataset_DeepGDel_NMA import create_gene_metabolite_dataset_nma
from baseline.graph_utils_nma import (
    load_metabolite_order_and_edges,
    metabolite_name_to_node_id,
    compute_nma,
)
from baseline.evaluations import calculate_metrics_for_val_metabolites_nma
from baseline.graphgdel_model.GeneLSTM import GeneLSTM
from baseline.graphgdel_model.MetaLSTM import MetaLSTM
from baseline.CombinedModel_DeepGDel_NMA import CombinedModel_DeepGDel_NMA


def parse_args():
    parser = argparse.ArgumentParser(description="NMA baseline: DeepGDel + Neighborhood Mean Aggregation")
    parser.add_argument("--CBM", type=str, default="e_coli_core", help="CBM model name")
    parser.add_argument("--use_cpu", type=int, choices=[0, 1], default=0, help="1 to force CPU")
    parser.add_argument("--split_seed", type=int, default=None, help="Legacy split seed (1001..1050)")
    parser.add_argument("--cv_protocol", choices=["legacy", "manuscript"], default="legacy",
                        help="manuscript: metabolite-held-out five-fold (needs cv_partitions.json)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index 0..4 for manuscript CV")
    parser.add_argument("--output_metrics_csv", type=str, default=None, help="If set, append metrics row to this CSV")
    parser.add_argument("--train", type=int, choices=[0, 1], default=0, help="1 to force re-training and save checkpoint")
    parser.add_argument("--train_epochs", type=int, default=50, help="Epochs when --train 1")
    return parser.parse_args()


def main():
    args = parse_args()
    CBM = args.CBM
    use_cpu = args.use_cpu
    print(f"Using CBM: {CBM}")

    device = torch.device("cpu" if use_cpu == 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    bound = 0.9
    N_FOLDS = 10
    FIRST_SPLIT_SEED = 1001
    vocab_embedding_dim = 32
    lstm_hidden_dim = 64
    batch_size = 8
    epochs = args.train_epochs if getattr(args, "train", 0) else 1
    random_seed = 1001
    random.seed(random_seed)
    np.random.seed(random_seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    gene_folder_path = os.path.join(main_dir, "Data", CBM, "Gene_AA")
    metabolite_smiles_path = os.path.join(main_dir, "Data", CBM, "Target metabolites_smile.csv")
    relationship_folder_path = os.path.join(main_dir, "Data", CBM, "Label")

    gene_names, gene_sequences_raw = read_gene_sequences(gene_folder_path)
    metabolite_names, smiles_features_raw = read_metabolite_smiles(metabolite_smiles_path)
    relationships = read_gene_metabolite_relationships(relationship_folder_path)
    metabolite_names = np.array(metabolite_names)

    unique_metabolites = list(relationships.keys())
    train_val_pool = list(unique_metabolites)
    metabolite_labels = np.array([
        1 if np.mean(relationships[meta][1]) >= 0.5 else 0 for meta in train_val_pool
    ])
    if args.cv_protocol == "manuscript":
        from utils.manuscript_cv import train_val_metabolites

        train_metabolites, val_metabolites = train_val_metabolites(CBM, args.fold, Path(main_dir))
        args.split_seed = 42 + args.fold
    else:
        split_seed = getattr(args, "split_seed", None) or FIRST_SPLIT_SEED
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
    eval_metabolites = val_metabolites if evaluate_fold_holdout else unique_metabolites
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

    metabolite_names_order, edge_index, num_nodes = load_metabolite_order_and_edges(
        main_dir, CBM, unique_metabolites=unique_metabolites
    )

    gene_count_all = {}
    gene_total_all = {}
    for meta_name in unique_metabolites:
        gnames, labels = relationships[meta_name]
        for g, label in zip(gnames, labels):
            if g not in gene_count_all:
                gene_count_all[g] = {"0": 0, "1": 0}
                gene_total_all[g] = 0
            gene_count_all[g][str(label)] += 1
            gene_total_all[g] += 1
    genes_to_exclude = []
    for g in gene_count_all:
        total = gene_total_all[g]
        p0 = gene_count_all[g]["0"] / total if total > 0 else 0
        p1 = gene_count_all[g]["1"] / total if total > 0 else 0
        if p0 >= bound or p1 >= bound:
            genes_to_exclude.append(g)

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(gene_sequences_raw)
    gene_vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(s) for s in gene_sequences_raw)
    gene_sequences = pad_sequences(
        tokenizer.texts_to_sequences(gene_sequences_raw), maxlen=max_len
    )
    smiles_tokenizer = Tokenizer(char_level=True)
    smiles_tokenizer.fit_on_texts(smiles_features_raw)
    smiles_vocab_size = len(smiles_tokenizer.word_index) + 1
    smiles_max_len = max(len(s) for s in smiles_features_raw)
    smiles_features = pad_sequences(
        smiles_tokenizer.texts_to_sequences(smiles_features_raw), maxlen=smiles_max_len
    )

    gene_names_list = list(gene_names)

    def build_xy_meta(train_metabolites_list):
        X_genes, X_smiles, y_list, meta_indices_list = [], [], [], []
        for meta_name in train_metabolites_list:
            gnames, labels = relationships[meta_name]
            meta_smiles_idx = int(np.where(metabolite_names == meta_name)[0][0])
            node_id = metabolite_name_to_node_id(meta_name, metabolite_names_order)
            if node_id < 0:
                node_id = 0
            for gene_name, label in zip(gnames, labels):
                gene_index = gene_names_list.index(gene_name)
                X_genes.append(gene_sequences[gene_index])
                X_smiles.append(smiles_features[meta_smiles_idx])
                y_list.append(label)
                meta_indices_list.append(node_id)
        return (
            np.array(X_genes),
            np.array(X_smiles),
            np.array(y_list),
            np.array(meta_indices_list, dtype=np.int64),
        )

    X_genes_train, X_smiles_train, y_train, meta_train = build_xy_meta(train_metabolites)
    X_genes_val, X_smiles_val, y_val, meta_val = build_xy_meta(val_metabolites)

    train_dataset = create_gene_metabolite_dataset_nma(
        X_genes_train, X_smiles_train, y_train, meta_train
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_dataset = create_gene_metabolite_dataset_nma(
        X_genes_val, X_smiles_val, y_val, meta_val
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    gene_lstm = GeneLSTM(gene_vocab_size, vocab_embedding_dim, lstm_hidden_dim)
    smiles_lstm = MetaLSTM(smiles_vocab_size, vocab_embedding_dim, lstm_hidden_dim)
    model = CombinedModel_DeepGDel_NMA(gene_lstm, smiles_lstm, lstm_hidden_dim)

    ckpt_path = os.path.join(main_dir, "baseline", f"NMA_{CBM}.sav")
    do_train = getattr(args, "train", 0)
    if do_train:
        print(f"Re-training NMA for {epochs} epochs and saving to {ckpt_path}")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        bce = nn.BCELoss()
        for epoch in range(epochs):
            model.train()
            all_smiles_for_z = []
            for name in metabolite_names_order:
                idx = np.where(metabolite_names == name)[0]
                if len(idx) > 0:
                    row = smiles_features[idx[0]]
                    all_smiles_for_z.append(row)
                else:
                    all_smiles_for_z.append(np.zeros(smiles_max_len, dtype=np.int64))
            smiles_tensor = torch.tensor(np.array(all_smiles_for_z), dtype=torch.long, device=device)
            with torch.no_grad():
                z_meta, _ = model.smiles_model(smiles_tensor)
                Z_meta_nma = compute_nma(z_meta, edge_index, num_nodes, no_neighbor_use_self=True)
            for batch in train_loader:
                gene_seq, smiles_seq, meta_indices, labels = [x.to(device) for x in batch]
                smiles_seq = smiles_seq.long()
                output, _, _ = model(gene_seq, smiles_seq, meta_indices, Z_meta_nma)
                loss = bce(output.view(-1), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print("Saved NMA checkpoint to", ckpt_path)
    else:
        model.to(device)
        if os.path.isfile(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state, strict=False)
                print("NMA model loaded from", ckpt_path)
            except Exception as e:
                print(
                    f"Could not load NMA checkpoint from {ckpt_path} ({e}). "
                    "Using randomly initialized weights."
                )
        else:
            print(
                f"No checkpoint at {ckpt_path}; using randomly initialized weights."
            )

    model.eval()
    with torch.no_grad():
        all_smiles_for_z = []
        for name in metabolite_names_order:
            idx = np.where(metabolite_names == name)[0]
            if len(idx) > 0:
                row = smiles_features[idx[0]]
                all_smiles_for_z.append(row)
            else:
                all_smiles_for_z.append(np.zeros(smiles_max_len, dtype=np.int64))
        smiles_tensor = torch.tensor(np.array(all_smiles_for_z), dtype=torch.long, device=device)
        z_meta, _ = model.smiles_model(smiles_tensor)
        Z_meta_nma = compute_nma(z_meta, edge_index, num_nodes, no_neighbor_use_self=True)

    summary = calculate_metrics_for_val_metabolites_nma(
        model, eval_metabolites, metabolite_names, metabolite_names_order,
        gene_sequences, smiles_features, relationships, device, genes_to_exclude, Z_meta_nma,
        train_metabolites=train_metabolites,
    )
    if args.output_metrics_csv and summary is not None:
        write_header = not os.path.exists(args.output_metrics_csv)
        with open(args.output_metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["CBM", "method", "split_seed", "accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc"])
            w.writerow([CBM, "NMA", args.split_seed] + [round(s, 4) for s in summary])


if __name__ == "__main__":
    main()
