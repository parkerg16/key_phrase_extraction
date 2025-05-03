import os
import json
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_graph_json(graph, path, labels):
    # For nodes, use the labels
    nodes = [
        {"id": n, "label": labels.get(n, str(n))} for n in graph.nodes()
    ]
    
    # For edges, include relationship type and weight if available
    edges = []
    for u, v, data in graph.edges(data=True):
        edge = {
            "source": u, 
            "target": v
        }
        
        # Add relationship type if available
        if 'relationship' in data:
            edge["relationship"] = data['relationship']
            
        # Add weight if available
        if 'weight' in data:
            edge["weight"] = data['weight']
            
        edges.append(edge)
    
    data = {
        "nodes": nodes,
        "edges": edges
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_plot(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot to {path}")
