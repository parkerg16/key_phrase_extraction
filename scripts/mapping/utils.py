import os
import json
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_graph_json(graph, path, labels):
    data = {
        "nodes": [
            {"id": n, "label": labels.get(n, str(n))} for n in graph.nodes()
        ],
        "edges": [
            {"source": u, "target": v} for u, v in graph.edges()
        ]
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_plot(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"📸 Saved plot to {path}")
