"""
Build metabolite graph from CBM JSON following the manuscript subsection
"Constructing Graph Representations of Metabolic Models" (4-step pipeline).
"""
import json
import os
import re
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Canonical currency metabolite set from manuscript (KEGG/BiGG)
DEFAULT_CURRENCY_LIST = [
    "atp", "adp", "amp", "nad", "nadh", "nadp", "nadph", "coa", "h", "h2o",
    "pi", "ppi", "co2", "nh4", "o2", "glutathione", "na1", "k", "mg2", "ca2",
]


def _base_metabolite_id(met_id):
    """Normalize metabolite id for currency matching: strip compartment suffix (_c, _e, etc.)."""
    if not met_id:
        return ""
    s = str(met_id).strip().lower()
    # Remove common compartment suffixes
    return re.sub(r"_[ce]$", "", re.sub(r"_[a-z0-9]+$", "", s))


def _parse_gpr(gene_reaction_rule):
    """Extract gene names from GPR string (and/or logic)."""
    genes = []
    and_conditions = (gene_reaction_rule or "").split(" and ")
    for and_cond in and_conditions:
        for gene in and_cond.split(" or "):
            gene = gene.strip().replace("(", "").replace(")", "").strip()
            if gene:
                genes.append(gene)
    return genes


def build_metabolite_graph_from_cbm(
    json_path,
    node_feature_path=None,
    top_k=10,
    currency_list=None,
    nodes_to_restore=None,
    unique_metabolites=None,
):
    """
    Four-step pipeline: tripartite -> G_0 -> G_1 (remove top-k) -> G_final (remove currency, restore).

    Returns:
        graph_data: PyG Data(x=x, edge_index=edge_index)
        metabolite_names_order: list of node names in graph order (same as x rows)
        num_nodes: int
    """
    if currency_list is None:
        currency_list = list(DEFAULT_CURRENCY_LIST)
    if nodes_to_restore is None:
        nodes_to_restore = []
    if unique_metabolites is None:
        unique_metabolites = []

    currency_set = {s.strip().lower() for s in currency_list}

    with open(json_path) as f:
        data = json.load(f)

    # Step 1: Tripartite graph
    G = nx.Graph()
    for reaction in data["reactions"]:
        reaction_id = reaction["id"]
        gene_reaction_rule = reaction.get("gene_reaction_rule", "")
        genes = _parse_gpr(gene_reaction_rule)
        metabolites = list(reaction.get("metabolites", {}).keys())

        G.add_node(reaction_id, bipartite=0, type="reaction")
        G.add_nodes_from(metabolites, bipartite=1, type="metabolite")
        for gene in genes:
            G.add_node(gene, bipartite=2, type="gene")
            G.add_edge(reaction_id, gene, bipartite=2)
        for met in metabolites:
            G.add_edge(reaction_id, met, bipartite=1)

    reactions_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}

    # Step 2: Induce metabolite graph G_0
    original_metabolite_graph = nx.Graph()
    for reaction_node in reactions_nodes:
        connected = [n for n in G.neighbors(reaction_node) if G.nodes[n].get("type") == "metabolite"]
        for m1, m2 in combinations(connected, 2):
            if original_metabolite_graph.has_edge(m1, m2):
                original_metabolite_graph[m1][m2]["weight"] = original_metabolite_graph[m1][m2].get("weight", 1) + 1
            else:
                original_metabolite_graph.add_edge(m1, m2, weight=1)

    # Step 3: Remove top-k by degree
    node_degrees = list(original_metabolite_graph.degree())
    df_degrees = pd.DataFrame(node_degrees, columns=["metabolite", "degree"])
    df_degrees = df_degrees.sort_values("degree", ascending=False)
    top_k_nodes = df_degrees.head(top_k)["metabolite"].tolist()

    removed_metabolite_graph = original_metabolite_graph.copy()
    removed_metabolite_graph.remove_nodes_from(top_k_nodes)

    # Step 4: Knowledge-based refinement -> G_final
    final_metabolite_graph = removed_metabolite_graph.copy()

    # (1) Remove canonical currency metabolites (match by base id)
    nodes_to_delete = [
        n for n in list(final_metabolite_graph.nodes())
        if _base_metabolite_id(n) in currency_set
    ]
    final_metabolite_graph.remove_nodes_from(nodes_to_delete)

    # (2) Restore manually selected nodes
    for node in nodes_to_restore:
        if node not in original_metabolite_graph:
            continue
        if node not in final_metabolite_graph:
            final_metabolite_graph.add_node(node)
        for neighbor in original_metabolite_graph.neighbors(node):
            if neighbor in final_metabolite_graph:
                final_metabolite_graph.add_edge(node, neighbor)

    # (3) Restore GCP / unique_metabolites
    for node in unique_metabolites:
        if node not in final_metabolite_graph and node in original_metabolite_graph:
            final_metabolite_graph.add_node(node)
            for neighbor in original_metabolite_graph.neighbors(node):
                if neighbor in final_metabolite_graph:
                    final_metabolite_graph.add_edge(node, neighbor)

    # Output: fixed node order (sorted for determinism), edge_index, node features
    graph_node_list = sorted(final_metabolite_graph.nodes())
    name_to_idx = {name: i for i, name in enumerate(graph_node_list)}
    num_nodes = len(graph_node_list)

    edge_list = list(final_metabolite_graph.edges())
    if not edge_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        rows = []
        for u, v in edge_list:
            i, j = name_to_idx[u], name_to_idx[v]
            rows.append([i, j])
        edge_index = torch.tensor(rows, dtype=torch.long).T.contiguous()

    # Node features: align CSV to graph_node_list or fallback to randn
    if node_feature_path and os.path.isfile(node_feature_path):
        try:
            df = pd.read_csv(node_feature_path)
            if "metabolite" not in df.columns:
                raise ValueError("CSV must have 'metabolite' column")
            csv_met_to_row = df.set_index("metabolite").to_dict("index")
            feat_cols = [c for c in df.columns if c != "metabolite"]
            feat_dim = len(feat_cols)
            x_rows = []
            for name in graph_node_list:
                if name in csv_met_to_row:
                    row = csv_met_to_row[name]
                    x_rows.append([row[c] for c in feat_cols])
                else:
                    x_rows.append([0.0] * feat_dim)
            x = torch.tensor(x_rows, dtype=torch.float32)
        except Exception:
            x = torch.randn(num_nodes, 64, dtype=torch.float32)
    else:
        x = torch.randn(num_nodes, 64, dtype=torch.float32)

    graph_data = Data(x=x, edge_index=edge_index)
    return graph_data, graph_node_list, num_nodes


def load_graph_data(graph_path, CBM, data_dir=None, unique_metabolites=None, **build_kwargs):
    """
    Load graph: if Data/{CBM}/{CBM}.json exists, run 4-step pipeline; else random graph from CSV.

    Returns:
        graph_data: PyG Data
        num_nodes: int
        metabolite_names_order: list or None. If from JSON pipeline, ordered node list; else None.
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(graph_path)))
    json_path = os.path.join(data_dir, CBM, f"{CBM}.json")

    if os.path.isfile(json_path):
        node_feature_path = graph_path
        if unique_metabolites is None:
            unique_metabolites = []
        graph_data, metabolite_names_order, num_nodes = build_metabolite_graph_from_cbm(
            json_path,
            node_feature_path=node_feature_path,
            unique_metabolites=unique_metabolites,
            **build_kwargs,
        )
        return graph_data, num_nodes, metabolite_names_order

    # Fallback: random graph when CBM JSON is unavailable
    try:
        node_features_df = pd.read_csv(graph_path)
        num_nodes = len(node_features_df)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        from torch_geometric.utils import to_undirected
        edge_index = to_undirected(edge_index)
        node_features = torch.randn(num_nodes, 64, dtype=torch.float32)
        graph_data = Data(x=node_features, edge_index=edge_index)
        return graph_data, num_nodes, None
    except Exception as e:
        print(f"Error loading graph data: {e}")
        num_nodes = 1000
        from torch_geometric.utils import to_undirected
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_index = to_undirected(edge_index)
        graph_data = Data(x=torch.randn(num_nodes, 64), edge_index=edge_index)
        return graph_data, num_nodes, None
