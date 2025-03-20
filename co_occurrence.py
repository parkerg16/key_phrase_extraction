import os
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from colorama import Fore, init
import json

init(autoreset=True)

# Paths
key_phrases_path = 'key_phrases'
graph_output_path = 'graph_data'

# Create output directory if it doesn't exist
if not os.path.exists(graph_output_path):
    os.makedirs(graph_output_path)
    print(f"Created folder: {graph_output_path}")

def load_keyphrases():
    """Load extracted keyphrases from files"""
    keyphrase_files = sorted([f for f in os.listdir(key_phrases_path) if f.endswith('.txt')])
    all_keyphrases = {}
    
    for file_name in keyphrase_files:
        chapter_id = int(file_name.split('_')[1])
        keyphrases = []
        
        file_path = os.path.join(key_phrases_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    phrase, score = line.strip().split(': ')
                    keyphrases.append((phrase, float(score)))
        
        all_keyphrases[chapter_id] = keyphrases
    
    return all_keyphrases

def build_co_occurrence_matrix(keyphrases_by_chapter):
    """Build co-occurrence matrix for keyphrases"""
    # Extract all unique keyphrases
    all_unique_keyphrases = set()
    for chapter_keyphrases in keyphrases_by_chapter.values():
        for phrase, _ in chapter_keyphrases:
            all_unique_keyphrases.add(phrase)
    
    # Create a mapping from keyphrase to index
    keyphrase_to_idx = {phrase: idx for idx, phrase in enumerate(all_unique_keyphrases)}
    idx_to_keyphrase = {idx: phrase for phrase, idx in keyphrase_to_idx.items()}
    
    # Initialize co-occurrence matrix
    n_keyphrases = len(all_unique_keyphrases)
    co_occurrence = np.zeros((n_keyphrases, n_keyphrases))
    
    # Fill co-occurrence matrix
    for chapter_id, chapter_keyphrases in keyphrases_by_chapter.items():
        chapter_phrases = [phrase for phrase, _ in chapter_keyphrases]
        
        # Update co-occurrence counts
        for i, phrase1 in enumerate(chapter_phrases):
            for phrase2 in chapter_phrases[i+1:]:
                idx1 = keyphrase_to_idx[phrase1]
                idx2 = keyphrase_to_idx[phrase2]
                
                # Increment co-occurrence count
                co_occurrence[idx1, idx2] += 1
                co_occurrence[idx2, idx1] += 1
    
    return co_occurrence, keyphrase_to_idx, idx_to_keyphrase

def create_graph(co_occurrence, idx_to_keyphrase, threshold=0.0):
    """Create NetworkX graph from co-occurrence matrix"""
    G = nx.Graph()
    
    # Add nodes (keyphrases)
    for idx, phrase in idx_to_keyphrase.items():
        G.add_node(phrase)
    
    # Add edges (co-occurrences)
    n_keyphrases = co_occurrence.shape[0]
    for i in range(n_keyphrases):
        for j in range(i+1, n_keyphrases):
            weight = co_occurrence[i, j]
            if weight > threshold:
                G.add_edge(idx_to_keyphrase[i], idx_to_keyphrase[j], weight=weight)
    
    return G

def visualize_graph(G, output_path=None, max_nodes=100):
    """Visualize the graph with node sizes based on centrality"""
    # Limit to top nodes by degree for visualization
    if len(G.nodes) > max_nodes:
        degrees = dict(nx.degree(G))
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, _ in top_nodes]
        G = G.subgraph(top_node_names)
    
    # Compute centrality metrics
    centrality = nx.degree_centrality(G)
    
    # Set node sizes based on centrality
    node_sizes = [centrality[node] * 5000 for node in G.nodes]
    
    # Get edge weights
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges]
    
    # Create layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    plt.figure(figsize=(16, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    
    # Draw edges with width based on weight
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5)
    
    # Draw labels only for the most central nodes
    top_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
    top_nodes = [node for node, _ in top_centrality]
    labels = {node: node for node in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    plt.title("Keyphrase Co-occurrence Network")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_path}")
    
    plt.close()

def export_graph_for_gcn(G, output_path):
    """Export graph data for GCN processing"""
    # Create node mapping
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Create edge list
    edges = []
    edge_weights = []
    
    for u, v, data in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        edges.append([i, j])
        edge_weights.append(data['weight'])
    
    # Create node features (using node names for now)
    # In a real scenario, you would use embeddings or other features
    # This is a placeholder for demonstration
    node_features = [[1.0] * 10 for _ in range(len(nodes))]  # Simple placeholder features
    
    # Prepare data for export
    graph_data = {
        'nodes': nodes,
        'node_to_idx': node_to_idx,
        'edges': edges,
        'edge_weights': edge_weights,
        'node_features': node_features
    }
    
    # Export to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Graph data for GCN exported to {output_path}")

def main():
    print(Fore.GREEN + "Loading keyphrases...")
    keyphrases_by_chapter = load_keyphrases()
    
    print(Fore.GREEN + "Building co-occurrence matrix...")
    co_occurrence, keyphrase_to_idx, idx_to_keyphrase = build_co_occurrence_matrix(keyphrases_by_chapter)
    
    print(Fore.GREEN + "Creating graph...")
    G = create_graph(co_occurrence, idx_to_keyphrase, threshold=1.0)
    
    print(Fore.GREEN + f"Created graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Visualize the graph
    viz_output_path = os.path.join(graph_output_path, 'keyphrase_graph.png')
    print(Fore.GREEN + "Visualizing graph...")
    visualize_graph(G, output_path=viz_output_path)
    
    # Export graph data for GCN
    gcn_data_path = os.path.join(graph_output_path, 'gcn_graph_data.json')
    print(Fore.GREEN + "Exporting graph data for GCN...")
    export_graph_for_gcn(G, gcn_data_path)
    
    print(Fore.RED + "Co-occurrence analysis complete!")

if __name__ == "__main__":
    main()