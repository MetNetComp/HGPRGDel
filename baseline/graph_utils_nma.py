"""
Graph utilities for NMA (Neighborhood Mean Aggregation) baseline.
Uses the GraphGDel metabolite pairing graph (utils.build_metabolite_graph.load_graph_data),
not the HGPRGDel compound hypergraph.
"""
import os
import pandas as pd
import torch


def load_metabolite_order_and_edges(main_dir, CBM, unique_metabolites=None):
    """
    Load metabolite order and edge_index using the same loader as GraphGdel
    (utils.build_metabolite_graph.load_graph_data). When CBM JSON exists, returns
    the manuscript-aligned graph and node order; otherwise random graph and node order from CSV.

    Returns:
        metabolite_names_order: list of metabolite names (graph node order).
        edge_index: torch.LongTensor (2, num_edges).
        num_nodes: int.
    """
    graph_path = os.path.join(main_dir, "Data", CBM, "SMILES_node_feature_final.csv")
    if not os.path.isfile(graph_path):
        raise FileNotFoundError(f"Graph node list not found: {graph_path}")

    data_dir = os.path.join(main_dir, "Data")
    try:
        from utils.build_metabolite_graph import load_graph_data
        graph_data, num_nodes, metabolite_names_order = load_graph_data(
            graph_path,
            CBM,
            data_dir=data_dir,
            unique_metabolites=unique_metabolites or [],
            top_k=10,
            nodes_to_restore=["pyr_c"],
        )
        edge_index = graph_data.edge_index
        if metabolite_names_order is None:
            df = pd.read_csv(graph_path)
            if "metabolite" in df.columns:
                metabolite_names_order = df["metabolite"].astype(str).tolist()
                num_nodes = len(metabolite_names_order)
            else:
                metabolite_names_order = list(range(num_nodes))
    except Exception as e:
        print(f"Warning: NMA graph load failed ({e}). Using CSV order and random edges.")
        df = pd.read_csv(graph_path)
        metabolite_names_order = df["metabolite"].astype(str).tolist() if "metabolite" in df.columns else list(range(len(df)))
        num_nodes = len(metabolite_names_order)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        from torch_geometric.utils import to_undirected
        edge_index = to_undirected(edge_index)

    return metabolite_names_order, edge_index, num_nodes


def metabolite_name_to_node_id(metabolite_name, metabolite_names_order):
    """Return node_id (index) for a metabolite name. -1 if not found."""
    try:
        return metabolite_names_order.index(metabolite_name)
    except ValueError:
        return -1


def compute_nma(features, edge_index, num_nodes, no_neighbor_use_self=True):
    """
    Neighborhood Mean Aggregation: for each node i, set
    out[i] = mean(features[j] for j in neighbors(i)).
    If node i has no neighbors, use no_neighbor_use_self: if True use features[i], else zeros.
    """
    device = features.device
    feat_dim = features.size(1)
    out = torch.zeros(num_nodes, feat_dim, device=device, dtype=features.dtype)

    if edge_index is not None and edge_index.numel() > 0:
        edge_index = edge_index.to(device)
    if edge_index is None or edge_index.numel() == 0:
        if no_neighbor_use_self:
            return features.clone()
        return out

    src, tgt = edge_index[0], edge_index[1]
    out.scatter_add_(0, tgt.unsqueeze(1).expand(-1, feat_dim), features[src])
    count = torch.zeros(num_nodes, device=device, dtype=features.dtype)
    count.scatter_add_(0, tgt, torch.ones_like(tgt, dtype=features.dtype))
    mask_no_neighbor = (count == 0)
    count = count.clamp(min=1)
    out = out / count.unsqueeze(1)
    if no_neighbor_use_self:
        out = torch.where(mask_no_neighbor.unsqueeze(1), features, out)
    return out
