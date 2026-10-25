#!/usr/bin/env python3
"""
HGPRGDel Main Training Script
=============================

Supports three GEMs: e_coli_core, iML1515, iMM904.
Model: ESM-2 + ChemBERTa + Hypergraph + GPR.
"""

import os
import sys
import json
import yaml

import torch
import argparse
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.paths import CHECKPOINT_EXT, checkpoint_filename
from utils.prediction_threshold import (
    BINARY_THRESHOLD_TRAIN_MACRO_F1,
    BINARY_THRESHOLD_YOUDEN,
    DEFAULT_BINARY_THRESHOLD_MODE,
)
from model import (
    HGPRGDelHypergraphModel,
    MetabolicDataset,
    BalancedMetabolicDataset,
    load_metabolic_data_for_gem,
    write_metabolite_gene_predictions_csv,
    get_evaluation_genes,
    train_epoch,
    validate_epoch_with_metabolite_thresholds,
    create_metabolite_based_cross_validation,
    build_fold_structure,
)


SUPPORTED_GEMS = ['e_coli_core', 'iML1515', 'iMM904']


def train_hgprgdel_cv(model, dataset, gene_sequences, metabolite_smiles, gene_names, metabolite_names,
                      reaction_compounds, reaction_genes, compound_reaction_edges, reaction_gene_edges,
                      gpr_rules, gene_to_idx, incidence_matrix, num_folds, batch_size, num_epochs,
                      learning_rate, device, evaluation_genes, reconstruction_weight, experiment_dir=None,
                      seed=42, early_stopping_patience=None, model_kwargs=None,
                      gene_backbone_features=None, metabolite_backbone_features=None,
                      target_to_compound_idx=None,
                      dataloader_num_workers=None,
                      use_amp: bool = True,
                      grad_clip_norm: float | None = None,
                      full_train_min_epochs: int = 0,
                      cbm: str | None = None,
                      batch_size_for_progress: int | None = None,
                      full_train: bool = False):
    """Train HGPRGDel with metabolite-based CV, or a single full-GEM pass when ``full_train``.

    With ``full_train=True``, all target metabolites are used for training; early stopping
    monitors train macro-F1 (no held-out validation fold). Model needs full data + batch indices.
    If experiment_dir is set, saves splits, checkpoints, and metrics there."""
    import torch.nn as nn
    from tqdm import tqdm

    from model.gpr_module import (
        infer_num_reactions,
        parse_gpr_rules_list,
        pad_gpr_parsed,
        build_gene_compound_reaction_incidence,
    )
    from utils.training_progress import TrainingProgressTracker

    progress = None
    if experiment_dir:
        progress = TrainingProgressTracker(
            experiment_dir,
            num_folds=1 if full_train else num_folds,
            num_epochs=num_epochs,
            cbm=cbm,
            batch_size=batch_size_for_progress or batch_size,
        )

    # O(1) name lookups (list.index per batch is prohibitive on iML1515-scale gene lists).
    metabolite_to_idx = {m: i for i, m in enumerate(metabolite_names)}
    if dataloader_num_workers is None:
        if getattr(device, "type", str(device)) == "cuda":
            dataloader_num_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
        else:
            dataloader_num_workers = 0
    pin_memory = getattr(device, "type", str(device)) == "cuda"
    dl_kw = dict(
        num_workers=dataloader_num_workers,
        pin_memory=pin_memory,
    )
    if dataloader_num_workers > 0:
        dl_kw["persistent_workers"] = True

    if full_train:
        all_metas = sorted({m for _, m in dataset.pairs})
        train_idx = list(range(len(dataset)))
        val_idx: list[int] = []
        train_metas = all_metas
        val_metas: list[str] = []
        fold_splits = [(train_idx, val_idx, train_metas, val_metas)]
        effective_num_folds = 1
        print(f"\nFull-GEM training on all {len(train_metas)} target metabolites "
              f"({len(train_idx)} gene–metabolite pairs)")
    else:
        fold_splits = list(create_metabolite_based_cross_validation(dataset, num_folds=num_folds, seed=seed))
        effective_num_folds = num_folds

    fold_results = []
    eval_gene_set = None if evaluation_genes is None else set(evaluation_genes)

    use_amp_effective = bool(use_amp and getattr(device, "type", str(device)) == "cuda")

    for fold, (train_idx, val_idx, train_metas, val_metas) in enumerate(fold_splits):
        fold_label = "full_train" if full_train else f"Fold {fold + 1}/{effective_num_folds}"
        print(f"\n{fold_label}")
        if progress is not None:
            progress.start_fold(fold, train_metas, val_metas)
        fold_structure = build_fold_structure(
            train_metas, metabolite_names, reaction_compounds, reaction_genes,
            compound_reaction_edges, reaction_gene_edges, gpr_rules, incidence_matrix,
            target_to_compound_idx=target_to_compound_idx
        )
        fold_reaction_compounds = fold_structure["reaction_compounds"]
        fold_reaction_genes = fold_structure["reaction_genes"]
        fold_cr_edges = fold_structure["compound_reaction_edges"]
        fold_rg_edges = fold_structure["reaction_gene_edges"]
        fold_gpr_rules = fold_structure["gpr_rules"]
        train_sub = torch.utils.data.Subset(dataset, train_idx)
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, **dl_kw)
        val_loader = None
        if val_idx:
            val_sub = torch.utils.data.Subset(dataset, val_idx)
            val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, **dl_kw)

        _kwargs = model_kwargs or {}
        fold_model = HGPRGDelHypergraphModel(**_kwargs)
        fold_model.to(device)
        opt = torch.optim.Adam(fold_model.parameters(), lr=learning_rate)
        crit = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp_effective)
        inc = fold_structure["incidence_matrix"].to(device)
        best_f1 = 0.0
        best_state = None

        gpr_parsed = None
        g_rxn = c_rxn = None
        if getattr(fold_model, "use_gpr", True):
            num_rxn_fold = infer_num_reactions(
                fold_reaction_compounds, fold_reaction_genes, fold_gpr_rules
            )
            gpr_parsed = pad_gpr_parsed(
                parse_gpr_rules_list(fold_gpr_rules, fold_model.gpr_parser),
                num_rxn_fold,
            )
            g_rxn, c_rxn = build_gene_compound_reaction_incidence(
                fold_reaction_genes,
                fold_reaction_compounds,
                len(gene_names),
                len(metabolite_smiles),
                num_rxn_fold,
                device,
            )

        from sklearn.metrics import accuracy_score, f1_score
        from utils.prediction_threshold import select_global_threshold_macro_f1, scores_to_binary
        from utils.training_progress import BATCH_PROGRESS_MIN_INTERVAL_SEC

        def run_epoch(loader, train_mode=True, epoch_idx: int = 0):
            import time

            if train_mode:
                fold_model.train()
            else:
                fold_model.eval()
            total_loss = 0.0
            loss_batches = 0
            preds, lbls = [], []
            total_batches = len(loader)
            last_progress_write = 0.0
            loss_ema: float | None = None
            for batch_idx, batch in enumerate(tqdm(loader, desc="Train" if train_mode else "Val", leave=False)):
                g_idx = [gene_to_idx[n] for n in batch['gene_name']]
                if 'compound_idx' in batch:
                    m_idx = [int(v) for v in batch['compound_idx']]
                elif target_to_compound_idx is not None:
                    m_idx = [target_to_compound_idx[n] for n in batch['metabolite_name']]
                else:
                    m_idx = [metabolite_to_idx[n] for n in batch['metabolite_name']]
                labels = batch['label'].float().to(device)
                if train_mode:
                    opt.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=use_amp_effective):
                        out = fold_model(
                            gene_sequences, metabolite_smiles,
                            fold_reaction_compounds, fold_reaction_genes, fold_cr_edges, fold_rg_edges,
                            incidence_matrix=inc, gpr_rules=fold_gpr_rules, gene_to_idx=gene_to_idx,
                            gpr_parsed=gpr_parsed,
                            gene_rxn_incidence=g_rxn,
                            compound_rxn_incidence=c_rxn,
                            batch_gene_idx=g_idx, batch_metabolite_idx=m_idx,
                            return_reconstruction_loss=False,
                            gene_backbone_features=gene_backbone_features,
                            metabolite_backbone_features=metabolite_backbone_features,
                        )
                    if out.dim() > 1:
                        out = out.squeeze()
                    loss = crit(out.float(), labels)
                    if not torch.isfinite(loss):
                        continue
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            fold_model.parameters(), grad_clip_norm
                        )
                    scaler.step(opt)
                    scaler.update()
                else:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=use_amp_effective):
                            out = fold_model(
                                gene_sequences, metabolite_smiles,
                                fold_reaction_compounds, fold_reaction_genes, fold_cr_edges, fold_rg_edges,
                                incidence_matrix=inc, gpr_rules=fold_gpr_rules, gene_to_idx=gene_to_idx,
                                gpr_parsed=gpr_parsed,
                                gene_rxn_incidence=g_rxn,
                                compound_rxn_incidence=c_rxn,
                                batch_gene_idx=g_idx, batch_metabolite_idx=m_idx,
                                return_reconstruction_loss=False,
                                gene_backbone_features=gene_backbone_features,
                                metabolite_backbone_features=metabolite_backbone_features,
                            )
                    if out.dim() > 1:
                        out = out.squeeze()
                    loss = crit(out.float(), labels)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    loss_batches += 1
                out_np = torch.sigmoid(out).detach().float().cpu().numpy()
                lbl_np = labels.cpu().numpy()
                batch_genes = list(batch['gene_name'])
                if out_np.ndim == 0:
                    out_np = out_np.reshape(1)
                    lbl_np = lbl_np.reshape(1)
                for i, gene_name in enumerate(batch_genes):
                    if train_mode or eval_gene_set is None or gene_name in eval_gene_set:
                        preds.append(float(out_np[i]))
                        lbls.append(float(lbl_np[i]))
                if (
                    train_mode
                    and progress is not None
                    and total_batches > 0
                ):
                    loss_val = float(loss.item())
                    loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val
                    now = time.monotonic()
                    at_end = batch_idx + 1 >= total_batches
                    if (
                        batch_idx == 0
                        or at_end
                        or now - last_progress_write >= BATCH_PROGRESS_MIN_INTERVAL_SEC
                    ):
                        progress.update_training_batch(
                            fold,
                            epoch_idx,
                            batch_idx,
                            total_batches,
                            phase="train",
                            loss_ema=loss_ema,
                        )
                        last_progress_write = now
            denom = max(1, loss_batches)
            mean_loss = total_loss / denom if loss_batches > 0 else float("nan")
            return mean_loss, preds, lbls

        def metrics_with_train_threshold(train_preds, train_lbls, val_preds, val_lbls):
            if not train_lbls:
                return 0.0, 0.0
            threshold = select_global_threshold_macro_f1(train_lbls, train_preds)
            train_bin = scores_to_binary(train_preds, threshold)
            val_bin = scores_to_binary(val_preds, threshold) if val_lbls else []
            train_acc = accuracy_score(train_lbls, train_bin)
            train_f1 = f1_score(train_lbls, train_bin, zero_division=0, average='macro')
            val_acc = accuracy_score(val_lbls, val_bin) if val_lbls else 0.0
            val_f1 = f1_score(val_lbls, val_bin, zero_division=0, average='macro') if val_lbls else 0.0
            return train_acc, train_f1, val_acc, val_f1

        epochs_no_improve = 0
        min_epochs_before_stop = int(full_train_min_epochs) if full_train else 0
        current_ep = 0
        epochs_run = 0
        try:
            for ep in range(num_epochs):
                current_ep = ep
                epochs_run = ep + 1
                train_loss, train_preds, train_lbls = run_epoch(
                    train_loader, train_mode=True, epoch_idx=ep
                )
                if val_loader is not None:
                    val_loss, val_preds, val_lbls = run_epoch(
                        val_loader, train_mode=False, epoch_idx=ep
                    )
                else:
                    val_loss, val_preds, val_lbls = 0.0, [], []
                train_acc, train_f1, val_acc, val_f1 = metrics_with_train_threshold(
                    train_preds, train_lbls, val_preds, val_lbls
                )
                selection_f1 = val_f1 if val_lbls else train_f1
                selection_label = "val_f1" if val_lbls else "train_f1"
                train_loss_s = (
                    f"{train_loss:.4f}"
                    if train_loss == train_loss  # noqa: PLR0124 — NaN check
                    else "nan"
                )
                val_loss_s = (
                    f"{val_loss:.4f}"
                    if val_loss == val_loss
                    else "nan"
                )
                print(f"  Epoch {ep+1}: train_loss={train_loss_s} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
                      f"val_loss={val_loss_s} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
                      f"best_{selection_label}={best_f1:.4f}")
                if selection_f1 > best_f1:
                    best_f1 = selection_f1
                    best_state = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
                    epochs_no_improve = 0
                elif (ep + 1) >= min_epochs_before_stop:
                    epochs_no_improve += 1
                if progress is not None:
                    progress.update_epoch(
                        fold,
                        ep,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        train_f1=train_f1,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        val_f1=val_f1,
                        best_f1=best_f1,
                        epochs_no_improve=epochs_no_improve,
                    )
                if (
                    early_stopping_patience is not None
                    and (ep + 1) >= min_epochs_before_stop
                    and epochs_no_improve >= early_stopping_patience
                ):
                    print(
                        f"  Early stopping at epoch {ep+1} "
                        f"(no improvement for {early_stopping_patience} epochs"
                        f"{f' after epoch {min_epochs_before_stop}' if min_epochs_before_stop else ''})"
                    )
                    break
        except KeyboardInterrupt:
            if progress is not None:
                progress.mark_interrupted(fold, current_ep)
            if experiment_dir:
                ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                interrupt_path = os.path.join(
                    ckpt_dir, checkpoint_filename(f'fold_{fold}_interrupt_epoch{current_ep}')
                )
                torch.save(
                    {k: v.cpu().clone() for k, v in fold_model.state_dict().items()},
                    interrupt_path,
                )
                print(f"\nKeyboardInterrupt: saved in-progress weights to {interrupt_path}")
                if best_state is not None:
                    best_path = os.path.join(
                        ckpt_dir,
                        checkpoint_filename(f'fold_{fold}_interrupt_best_val_to_epoch{current_ep}'),
                    )
                    torch.save(best_state, best_path)
                    print(f"KeyboardInterrupt: saved best validation weights to {best_path}")
            raise
        except Exception as exc:
            if progress is not None:
                progress.mark_crashed(fold, current_ep, exc)
            raise

        if experiment_dir:
            ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_stem = 'full_train_best' if full_train else f'fold_{fold}_best'
            pth = os.path.join(ckpt_dir, checkpoint_filename(ckpt_stem))
            split_dir = os.path.join(experiment_dir, 'splits')
            os.makedirs(split_dir, exist_ok=True)
            with open(os.path.join(split_dir, f'fold_{fold}_train_metabolites.json'), 'w') as f:
                json.dump(list(train_metas), f, indent=2)
            with open(os.path.join(split_dir, f'fold_{fold}_val_metabolites.json'), 'w') as f:
                json.dump(list(val_metas), f, indent=2)
        else:
            os.makedirs('models', exist_ok=True)
            pth = os.path.join('models', checkpoint_filename(f'hgprgdel_hypergraph_{fold}_best'))
        if best_state:
            torch.save(best_state, pth)
        if progress is not None:
            progress.complete_fold(
                fold,
                best_f1=best_f1,
                epochs_run=epochs_run,
                checkpoint=os.path.relpath(pth, experiment_dir) if experiment_dir else pth,
            )
        fold_results.append({'fold': fold + 1, 'best_f1': best_f1, 'train_metabolites': train_metas, 'val_metabolites': val_metas})
    if experiment_dir:
        metrics_path = os.path.join(experiment_dir, 'metrics.json')
        mean_f1 = sum(r['best_f1'] for r in fold_results) / len(fold_results) if fold_results else 0
        metrics_data = {
            'full_train': full_train,
            'fold_results': [{'fold': r['fold'], 'best_f1': r['best_f1']} for r in fold_results],
            'mean_f1': mean_f1,
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        if progress is not None:
            progress.mark_training_complete(mean_f1=mean_f1)
    return fold_results


def _load_or_create_backbone_cache(model, gene_sequences, metabolite_smiles, cache_dir, refresh=False):
    """Cache frozen language-model pooled hidden states before trainable projections."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'full_gem_language_backbone_features.pt')
    if os.path.exists(cache_path) and not refresh:
        print(f"Loading embedding cache: {cache_path}")
        cache = torch.load(cache_path, map_location='cpu')
        gene_features = cache.get('gene_backbone_features')
        metabolite_features = cache.get('metabolite_backbone_features')
        if metabolite_features is None or metabolite_features.size(0) == len(metabolite_smiles):
            return gene_features, metabolite_features
        print("Cached metabolite feature count does not match full-GEM inputs; rebuilding cache.")

    print(f"Creating embedding cache: {cache_path}")
    model.eval()
    with torch.no_grad():
        if hasattr(model.gene_encoder, 'extract_pooled_features'):
            gene_features = model.gene_encoder.extract_pooled_features(gene_sequences)
        else:
            gene_features = None
        if hasattr(model.metabolite_encoder, 'extract_pooled_features'):
            metabolite_features = model.metabolite_encoder.extract_pooled_features(metabolite_smiles)
        else:
            metabolite_features = None
    torch.save({
        'gene_backbone_features': gene_features,
        'metabolite_backbone_features': metabolite_features,
    }, cache_path)
    return gene_features, metabolite_features


PRESET_PATHS = {
    'default': None,
    'lightweight_fast': 'trainer/configs/lightweight_fast.yaml',
}


def _load_config(path_or_preset):
    """Load config from file path or preset name. Returns dict or None."""
    if path_or_preset is None:
        return None
    if path_or_preset in PRESET_PATHS:
        p = PRESET_PATHS[path_or_preset]
        if p is None:
            return None
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
    else:
        path = path_or_preset
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_config(args, config):
    """Override args with config. config can have model.* and training.* keys.

    Also accepts a flat experiment dump (e.g. ``config.yaml`` written under
    ``--experiment_dir``) with top-level ``batch_size``, ``model_kwargs``, etc.
    """
    if not config:
        return args, {}
    training = config.get('training')
    if not training:
        flat_training_keys = (
            'batch_size', 'num_epochs', 'num_folds', 'learning_rate', 'early_stopping_patience',
        )
        if any(k in config for k in flat_training_keys):
            training = {k: config[k] for k in flat_training_keys if k in config}
        else:
            training = {}
    if not isinstance(training, dict):
        training = {}

    model_cfg = config.get('model')
    if not model_cfg and isinstance(config.get('model_kwargs'), dict):
        model_cfg = config['model_kwargs']
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    for k, v in training.items():
        if hasattr(args, k):
            # Ensure numeric types (yaml may load 3e-4 as str in some cases)
            if k in ('batch_size', 'num_epochs', 'num_folds', 'early_stopping_patience') and v is not None:
                v = int(v)
            elif k in ('learning_rate',) and v is not None:
                v = float(v)
            setattr(args, k, v)
    model_kwargs = {
        k: v for k, v in model_cfg.items()
        if k in ('gene_dim', 'metabolite_dim', 'hypergraph_conv_dim', 'hypergraph_num_layers',
                 'gpr_assoc_dim', 'hidden_dim', 'dropout', 'use_hypergraph',
                 'use_gpr', 'gpr_aggregation', 'gpr_and_branch_mode', 'use_pretrained_encoders', 'use_nma')
    }
    early_patience = training.get('early_stopping_patience')
    if early_patience is None and 'early_stopping_patience' in config:
        early_patience = config['early_stopping_patience']
    if early_patience is not None:
        args.early_stopping_patience = int(early_patience)

    # Flat experiment dump: common CLI mirrors (omit experiment_dir so CLI output path wins)
    for k in ('balance_strategy', 'eval_threshold', 'binary_threshold', 'seed', 'reconstruction_weight', 'device', 'data_root', 'CBM'):
        if k in config and config[k] is not None and hasattr(args, k):
            setattr(args, k, config[k])
    if getattr(args, 'experiment_dir', None) is None and config.get('experiment_dir'):
        args.experiment_dir = str(config['experiment_dir'])
    if 'cache_dir' in config and config['cache_dir'] and getattr(args, 'cache_dir', None) is None:
        args.cache_dir = str(config['cache_dir'])
    if 'cache_embeddings' in config:
        args.cache_embeddings = bool(config['cache_embeddings'])
    if config.get('no_hypergraph'):
        args.no_hypergraph = True
    if config.get('no_gpr'):
        args.no_gpr = True
    if config.get('gpr_mean_pooling'):
        args.gpr_mean_pooling = True
    if config.get('gpr_and_branch_mode'):
        args.gpr_and_branch_mode = str(config['gpr_and_branch_mode'])
    if config.get('no_pretrained_encoders'):
        args.no_pretrained_encoders = True
    if config.get('nma_baseline'):
        args.nma_baseline = True
    if config.get('full_train'):
        args.full_train = True
    if config.get('no_amp'):
        args.no_amp = True
    if config.get('grad_clip_norm') is not None:
        args.grad_clip_norm = float(config['grad_clip_norm'])
    if config.get('full_train_min_epochs') is not None:
        args.full_train_min_epochs = int(config['full_train_min_epochs'])
    return args, model_kwargs


def main():
    parser = argparse.ArgumentParser(description='HGPRGDel: growth-coupled gene deletion prediction')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--preset', type=str, default='default', choices=list(PRESET_PATHS),
                        help='Preset: default | lightweight_fast')
    parser.add_argument('--data_root', type=str, default='Data',
                        help='Path to Data folder containing GEM subdirs')
    parser.add_argument('--CBM', type=str, default='e_coli_core', choices=SUPPORTED_GEMS,
                        help='GEM: e_coli_core, iML1515, iMM904')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument(
        '--full_train',
        action='store_true',
        help='Train on all target metabolites (no CV hold-out); early stopping uses train macro-F1',
    )
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument(
        '--eval_threshold',
        type=float,
        default=0.9,
        help='Exclude genes with >= this label-consistency fraction from evaluation metrics (default 0.9)',
    )
    parser.add_argument(
        '--binary_threshold',
        type=str,
        default=DEFAULT_BINARY_THRESHOLD_MODE,
        choices=[BINARY_THRESHOLD_TRAIN_MACRO_F1, BINARY_THRESHOLD_YOUDEN],
        help=(
            'Binarize retention scores: train_macro_f1 (default; global threshold tuned on '
            'training-metabolite pairs) or youden_per_metabolite (optional legacy per-metabolite ROC)'
        ),
    )
    parser.add_argument('--balance_strategy', type=str, default='oversample',
                        choices=['none', 'oversample', 'undersample', 'both'])
    parser.add_argument('--reconstruction_weight', type=float, default=0.0,
                        help='0 to disable reconstruction loss')
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='If set, all outputs (config, splits, checkpoints, metrics, predictions) go here')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help=f'Path to a {CHECKPOINT_EXT} state dict; skip CV training and only run full-data prediction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Stop training when val_f1 does not improve for N epochs')
    parser.add_argument('--cache_embeddings', action='store_true', default=True,
                        help='Precompute frozen ESM/ChemBERTa pooled hidden states before training')
    parser.add_argument('--no_cache_embeddings', action='store_false', dest='cache_embeddings',
                        help='Disable language-model feature caching')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for cached language-model features; default: <experiment_dir>/cache or .cache/<CBM>')
    parser.add_argument('--refresh_cache', action='store_true',
                        help='Recompute cached language-model features even if a cache file exists')
    parser.add_argument('--no_hypergraph', action='store_true',
                        help='Ablation: disable Graph-M hypergraph convolution')
    parser.add_argument(
        '--graph_encoder',
        type=str,
        default='hypergraph',
        choices=['hypergraph', 'bipartite', 'pairwise'],
        help='Graph-M encoder variant when hypergraph is enabled',
    )
    parser.add_argument('--no_gpr', action='store_true',
                        help='Ablation: disable GPR-M reaction context')
    parser.add_argument('--gpr_mean_pooling', action='store_true',
                        help='Ablation: replace SoftBooleanAggregator with mean pooling over GPR genes')
    parser.add_argument(
        '--gpr_and_branch_mode',
        type=str,
        default='combined',
        choices=['combined', 'mean_only', 'soft_min_only'],
        help='Ablation: which soft-AND branches to use inside SoftBooleanAggregator',
    )
    parser.add_argument('--no_pretrained_encoders', action='store_true',
                        help='Ablation: replace ESM/ChemBERTa with simple character encoders')
    parser.add_argument('--nma_baseline', action='store_true',
                        help='Use neighborhood-mean aggregation for compound features (ablation; not part of component ablation suite)')
    parser.add_argument(
        '--dataloader_num_workers',
        type=int,
        default=-1,
        help='DataLoader worker processes; -1=auto (CUDA: min(4, CPUs-1)), 0=load in main process only',
    )
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable CUDA automatic mixed precision (autocast + GradScaler)')
    parser.add_argument(
        '--grad_clip_norm',
        type=float,
        default=None,
        help='Clip gradient L2 norm after backward (e.g. 1.0); omit to disable',
    )
    parser.add_argument(
        '--full_train_min_epochs',
        type=int,
        default=0,
        help='Full_train only: do not count early-stopping patience until this epoch',
    )
    args = parser.parse_args()

    explicit_device = args.device

    # Load config: --config takes precedence over --preset
    config_path = args.config if args.config else (args.preset if args.preset != 'default' else None)
    config = _load_config(config_path)
    model_kwargs = {}
    root = os.path.dirname(os.path.abspath(__file__))
    nested = None
    if config and isinstance(config.get('config'), str):
        nested_path = str(config['config']).replace("\\", "/")
        nested = _load_config(nested_path)
        if nested is None:
            alt = os.path.join(root, nested_path)
            if os.path.isfile(alt):
                nested = _load_config(alt)
    if nested:
        args, model_kwargs = _apply_config(args, nested)
    if config:
        args, mk_dump = _apply_config(args, config)
        model_kwargs = {**model_kwargs, **mk_dump}
    elif not nested:
        args, model_kwargs = _apply_config(args, None)
    if explicit_device != 'auto':
        args.device = explicit_device
    if args.no_hypergraph:
        model_kwargs['use_hypergraph'] = False
    else:
        model_kwargs['graph_encoder_type'] = args.graph_encoder
    if args.no_gpr:
        model_kwargs['use_gpr'] = False
    if args.gpr_mean_pooling:
        model_kwargs['gpr_aggregation'] = 'mean_pooling'
    elif getattr(args, 'gpr_and_branch_mode', 'combined') != 'combined':
        model_kwargs['gpr_and_branch_mode'] = args.gpr_and_branch_mode
    if args.no_pretrained_encoders:
        model_kwargs['use_pretrained_encoders'] = False
        args.cache_embeddings = False
    if args.nma_baseline:
        model_kwargs['use_hypergraph'] = False
        model_kwargs['use_gpr'] = False
        model_kwargs['use_nma'] = True

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    data_path = os.path.join(args.data_root, args.CBM)

    if args.experiment_dir:
        os.makedirs(args.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(args.experiment_dir, 'logs'), exist_ok=True)
        config_data = {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in vars(args).items()}
        if model_kwargs:
            config_data['model_kwargs'] = model_kwargs
        with open(os.path.join(args.experiment_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    print(f"Loading {args.CBM} from {data_path}")
    result = load_metabolic_data_for_gem(args.data_root, args.CBM)
    (gene_sequences, metabolite_smiles, task_labels, gene_names, metabolite_names,
     reaction_compounds, reaction_genes, compound_reaction_edges, reaction_gene_edges,
     gpr_rules, gene_to_idx, incidence_matrix, _, _) = result[:14]
    compound_names = result[14] if len(result) > 14 else metabolite_names
    target_to_compound_idx = result[15] if len(result) > 15 else None

    if args.balance_strategy == 'none':
        dataset = MetabolicDataset(
            gene_sequences, metabolite_smiles, task_labels, gene_names, metabolite_names,
            reaction_compounds, reaction_genes, compound_reaction_edges, reaction_gene_edges,
            gpr_rules, gene_to_idx, compound_names=compound_names,
            target_to_compound_idx=target_to_compound_idx
        )
    else:
        dataset = BalancedMetabolicDataset(
            gene_sequences, metabolite_smiles, task_labels, gene_names, metabolite_names,
            reaction_compounds, reaction_genes, compound_reaction_edges, reaction_gene_edges,
            gpr_rules, gene_to_idx, balance_strategy=args.balance_strategy,
            compound_names=compound_names, target_to_compound_idx=target_to_compound_idx
        )

    evaluation_genes = get_evaluation_genes(gene_names, task_labels, metabolite_names, args.eval_threshold)

    model = HGPRGDelHypergraphModel(**(model_kwargs or {}))
    model.to(device)

    gene_backbone_features = None
    metabolite_backbone_features = None
    if args.cache_embeddings:
        cache_dir = args.cache_dir
        if cache_dir is None:
            cache_dir = os.path.join(args.experiment_dir, 'cache') if args.experiment_dir else os.path.join('.cache', args.CBM)
        gene_backbone_features, metabolite_backbone_features = _load_or_create_backbone_cache(
            model, gene_sequences, metabolite_smiles, cache_dir, refresh=args.refresh_cache
        )

    resume_ckpt = getattr(args, 'resume_checkpoint', None)
    if resume_ckpt:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f'--resume_checkpoint not found: {resume_ckpt}')
        state = torch.load(resume_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [ckpt] missing keys (random init): {missing}")
        if unexpected:
            print(f"  [ckpt] unexpected keys (ignored): {unexpected}")
        print(f"Loaded checkpoint (eval only): {resume_ckpt}")
        fold_results = []
    else:
        fold_results = train_hgprgdel_cv(
            model, dataset, gene_sequences, metabolite_smiles, gene_names, metabolite_names,
            reaction_compounds, reaction_genes, compound_reaction_edges, reaction_gene_edges,
            gpr_rules, gene_to_idx, incidence_matrix,
            num_folds=args.num_folds, batch_size=args.batch_size, num_epochs=args.num_epochs,
            learning_rate=args.learning_rate, device=device,
            evaluation_genes=evaluation_genes, reconstruction_weight=args.reconstruction_weight,
            experiment_dir=args.experiment_dir, seed=args.seed,
            early_stopping_patience=getattr(args, 'early_stopping_patience', None),
            model_kwargs=model_kwargs or None,
            gene_backbone_features=gene_backbone_features,
            metabolite_backbone_features=metabolite_backbone_features,
            target_to_compound_idx=target_to_compound_idx,
            dataloader_num_workers=(
                None if args.dataloader_num_workers < 0 else args.dataloader_num_workers
            ),
            use_amp=not args.no_amp,
            grad_clip_norm=getattr(args, 'grad_clip_norm', None),
            full_train_min_epochs=int(getattr(args, 'full_train_min_epochs', 0) or 0),
            cbm=args.CBM,
            batch_size_for_progress=args.batch_size,
            full_train=getattr(args, 'full_train', False),
        )

    if fold_results:
        f1s = [r['best_f1'] for r in fold_results]
        if getattr(args, 'full_train', False):
            print(f"\nFull-train best macro-F1 (train): {f1s[0]:.4f}")
        else:
            print(f"\nAverage F1: {sum(f1s)/len(f1s):.4f}")

        best = max(fold_results, key=lambda x: x['best_f1'])
        if args.experiment_dir:
            if getattr(args, 'full_train', False):
                best_path = os.path.join(
                    args.experiment_dir, 'checkpoints', checkpoint_filename('full_train_best'),
                )
            else:
                best_path = os.path.join(
                    args.experiment_dir,
                    'checkpoints',
                    checkpoint_filename(f'fold_{best["fold"]-1}_best'),
                )
            best_overall = os.path.join(
                args.experiment_dir, 'checkpoints', checkpoint_filename('best_overall'),
            )
            if os.path.exists(best_path):
                state = torch.load(best_path, map_location=device)
                model.load_state_dict(state, strict=False)
                torch.save(state, best_overall)
        else:
            best_path = os.path.join(
                'models', checkpoint_filename(f'hgprgdel_hypergraph_{best["fold"]-1}_best'),
            )
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
    elif resume_ckpt:
        print("Skipped cross-validation training; using checkpoint weights for inference.")

    pred_output_dir = None
    if args.experiment_dir:
        pred_output_dir = os.path.join(args.experiment_dir, 'predictions')
        os.makedirs(pred_output_dir, exist_ok=True)

    train_metabolites_for_threshold = None
    if fold_results:
        best_for_threshold = max(fold_results, key=lambda x: x['best_f1'])
        train_metabolites_for_threshold = best_for_threshold['train_metabolites']
    elif args.CBM:
        try:
            from utils.manuscript_cv import train_val_metabolites
            train_metabolites_for_threshold, _ = train_val_metabolites(args.CBM, 0)
        except FileNotFoundError:
            # Defensive fallback if utils/cv_partitions.json is missing.
            print(
                "Warning: utils/cv_partitions.json not found; "
                "tuning binary threshold on all target metabolites."
            )
            train_metabolites_for_threshold = list(metabolite_names)

    output_df, binary_preds, predictions = write_metabolite_gene_predictions_csv(
        model, gene_sequences, metabolite_smiles, gene_names, metabolite_names,
        task_labels, reaction_compounds, reaction_genes, compound_reaction_edges,
        reaction_gene_edges, data_path, device, gpr_rules=gpr_rules, gene_to_idx=gene_to_idx,
        incidence_matrix=incidence_matrix, output_dir=pred_output_dir,
        gene_backbone_features=gene_backbone_features,
        metabolite_backbone_features=metabolite_backbone_features,
        target_to_compound_idx=target_to_compound_idx,
        binary_threshold=args.binary_threshold,
        train_metabolites=train_metabolites_for_threshold,
    )

    if args.experiment_dir:
        print(f"Output: {pred_output_dir}/all_metabolites_predictions.csv")
        if not resume_ckpt and fold_results:
            from utils.training_progress import mark_experiment_completed

            mark_experiment_completed(args.experiment_dir)
    else:
        print(f"Output: {data_path}/Results/all_metabolites_predictions.csv")
    print("Done.")


if __name__ == "__main__":
    main()
