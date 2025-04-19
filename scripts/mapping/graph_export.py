# scripts/mapping/graph_export.py

import os
import argparse
import networkx as nx
from matplotlib import pyplot as plt
from utils import ensure_dir, save_graph_json, save_plot

def export_graph(G, labels, output_dir="exports", filename_prefix="concept_map", pos=None):
    ensure_dir(output_dir)

    # 1. Save as PNG image
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = pos or nx.spring_layout(G, k=1.5, seed=42)
    nx.draw(G, pos, with_labels=False, node_size=300, node_color="skyblue", edge_color="gray", arrows=True, ax=ax)
    nx.draw_networkx_labels(
        G, pos, labels,
        font_size=8,
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'),
        ax=ax
    )
    save_path = os.path.join(output_dir, f"{filename_prefix}.png")
    save_plot(fig, save_path)

    # 2. Save as JSON
    json_path = os.path.join(output_dir, f"{filename_prefix}.json")
    save_graph_json(G, json_path, labels)

def main():
    parser = argparse.ArgumentParser(description="Export graph visualization and structure")
    parser.add_argument("--output_dir", type=str, default="exports")
    args = parser.parse_args()

    # Demo example
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3)])
    labels = {0: "Machine Learning", 1: "Supervised", 2: "Unsupervised", 3: "Regression"}
    export_graph(G, labels, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
