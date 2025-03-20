import os
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from colorama import Fore, init
import json

init(autoreset=True)

# Paths
key_phrases_path = 'key_phrases_enhanced'  # Use the enhanced keyphrases
graph_output_path = 'graph_data_enhanced'

# Create output directory if it doesn't exist
if not os.path.exists(graph_output_path):
    os.makedirs(graph_output_path)
    print(f"Created folder: {graph_output_path}")

def load_keyphrases(min_score=0.15):
    """Load extracted keyphrases from files with score filtering"""
    keyphrase_files = sorted([f for f in os.listdir(key_phrases_path) if f.endswith('.txt')])
    if not keyphrase_files:
        raise FileNotFoundError(f"No keyphrase files found in {key_phrases_path}. "
                               f"Please run key_word_extraction_enhanced.py first.")
    
    all_keyphrases = {}
    
    for file_name in keyphrase_files:
        chapter_id = int(file_name.split('_')[1])
        keyphrases = []
        
        file_path = os.path.join(key_phrases_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    phrase, score_str = line.strip().split(': ')
                    score = float(score_str)
                    # Only keep keyphrases with scores above threshold
                    if score >= min_score:
                        keyphrases.append((phrase, score))
        
        all_keyphrases[chapter_id] = keyphrases
    
    print(f"Loaded keyphrases from {len(keyphrase_files)} chapters")
    total_keyphrases = sum(len(kps) for kps in all_keyphrases.values())
    print(f"Total keyphrases after filtering (score >= {min_score}): {total_keyphrases}")
    
    return all_keyphrases

def build_co_occurrence_matrix(keyphrases_by_chapter):
    """Build co-occurrence matrix for keyphrases with score weighting"""
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
    
    # Fill co-occurrence matrix with weighted connections
    for chapter_id, chapter_keyphrases in keyphrases_by_chapter.items():
        # Create a dictionary of phrases and scores for this chapter
        chapter_phrase_scores = {phrase: score for phrase, score in chapter_keyphrases}
        chapter_phrases = list(chapter_phrase_scores.keys())
        
        # Update co-occurrence counts with score weighting
        for i, phrase1 in enumerate(chapter_phrases):
            score1 = chapter_phrase_scores[phrase1]
            for phrase2 in chapter_phrases[i+1:]:
                score2 = chapter_phrase_scores[phrase2]
                idx1 = keyphrase_to_idx[phrase1]
                idx2 = keyphrase_to_idx[phrase2]
                
                # Weight by the product of the scores
                weight = score1 * score2
                
                # Increment co-occurrence count
                co_occurrence[idx1, idx2] += weight
                co_occurrence[idx2, idx1] += weight
    
    return co_occurrence, keyphrase_to_idx, idx_to_keyphrase

def create_graph(co_occurrence, idx_to_keyphrase, threshold=0.05):
    """Create NetworkX graph from co-occurrence matrix with threshold"""
    G = nx.Graph()
    
    # Add nodes (keyphrases)
    for idx, phrase in idx_to_keyphrase.items():
        G.add_node(phrase)
    
    # Add edges (co-occurrences)
    n_keyphrases = co_occurrence.shape[0]
    edges_added = 0
    
    for i in range(n_keyphrases):
        for j in range(i+1, n_keyphrases):
            weight = co_occurrence[i, j]
            if weight > threshold:
                G.add_edge(idx_to_keyphrase[i], idx_to_keyphrase[j], weight=weight)
                edges_added += 1
    
    print(f"Added {edges_added} edges with weight > {threshold}")
    
    return G

def visualize_graph(G, output_path=None, max_nodes=150):
    """Visualize the graph with node sizes based on centrality"""
    if len(G.nodes) == 0:
        print("Graph is empty. Nothing to visualize.")
        return
    
    # Limit to top nodes by degree for visualization
    if len(G.nodes) > max_nodes:
        print(f"Graph has {len(G.nodes)} nodes. Limiting visualization to top {max_nodes} nodes.")
        degrees = dict(nx.degree(G))
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, _ in top_nodes]
        G = G.subgraph(top_node_names)
    
    # Compute centrality metrics
    centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))
    
    # Combine centrality measures for node sizing
    combined_centrality = {}
    for node in G.nodes:
        combined_centrality[node] = centrality[node] + betweenness[node]
    
    # Set node sizes based on centrality
    node_sizes = [combined_centrality[node] * 5000 + 100 for node in G.nodes]
    
    # Get edge weights for thickness
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges]
    
    # Create layout (try different layouts)
    try:
        # Try force-directed layout with stronger repulsion
        pos = nx.spring_layout(G, k=0.4, iterations=100, seed=42)
    except:
        # Fallback to simpler layout
        pos = nx.kamada_kawai_layout(G)
    
    # Create figure
    plt.figure(figsize=(20, 16))
    
    # Draw nodes with a colormap based on centrality
    node_color = list(combined_centrality.values())
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color, 
                          cmap=plt.cm.viridis, alpha=0.8)
    
    # Draw edges with width based on weight
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='gray')
    
    # Draw labels only for the most central nodes (top 40)
    top_centrality = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)[:40]
    top_nodes = [node for node, _ in top_centrality]
    labels = {node: node for node in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
    
    # Add colorbar to show centrality scale
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(min(node_color), max(node_color)))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.7)
    cbar.set_label('Centrality', fontsize=14)
    
    plt.title("Enhanced Keyphrase Co-occurrence Network", fontsize=18)
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
        edge_weights.append(float(data['weight']))  # Ensure weights are native Python floats
    
    # Calculate node importance features
    centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))
    
    # Create richer node features
    node_features = []
    for node in nodes:
        # Use centrality measures and other features
        features = [
            centrality[node],
            betweenness[node],
            len(node.split()),  # Number of words in phrase
            1.0  # Bias term
        ]
        # Pad to 10 dimensions with zeros
        features.extend([0.0] * (10 - len(features)))
        node_features.append(features)
    
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

def output_graph_stats(G):
    """Output basic statistics about the graph"""
    print(Fore.GREEN + "Graph Statistics:")
    print(f"Number of nodes (keyphrases): {len(G.nodes())}")
    print(f"Number of edges (co-occurrences): {len(G.edges())}")
    
    if len(G.nodes()) > 0:
        # Calculate degree for each node
        degrees = dict(nx.degree(G))
        
        # Find the nodes with highest degree (most connections)
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:30]
        
        print(Fore.GREEN + "\nTop 30 keyphrases by connections:")
        for node, degree in top_nodes:
            print(f"  {node}: {degree} connections")
        
        # Calculate betweenness centrality for key nodes
        betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
        
        # Find the nodes with highest betweenness centrality
        top_central_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
        
        print(Fore.GREEN + "\nTop 20 keyphrases by centrality (bridge concepts):")
        for node, centrality in top_central_nodes:
            print(f"  {node}: {centrality:.4f} centrality")
        
        # Find communities in the graph
        try:
            communities = nx.community.greedy_modularity_communities(G)
            print(Fore.GREEN + f"\nDetected {len(communities)} distinct concept communities")
            
            # Print the top 5 communities with their most central nodes
            for i, community in enumerate(list(communities)[:5]):
                print(f"Community {i+1} ({len(community)} nodes):")
                # For each community, find the most central nodes
                community_graph = G.subgraph(community)
                community_centrality = nx.degree_centrality(community_graph)
                top_community_nodes = sorted(community_centrality.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:5]
                for node, cent in top_community_nodes:
                    print(f"    {node}")
        except:
            print("Could not detect communities - graph may be too sparse or disconnected")

def main():
    print(Fore.GREEN + "Loading enhanced keyphrases...")
    try:
        keyphrases_by_chapter = load_keyphrases(min_score=0.15)  # Adjust min_score if needed
    except FileNotFoundError as e:
        print(Fore.RED + str(e))
        return
    
    print(Fore.GREEN + "Building co-occurrence matrix...")
    co_occurrence, keyphrase_to_idx, idx_to_keyphrase = build_co_occurrence_matrix(keyphrases_by_chapter)
    
    print(Fore.GREEN + "Creating graph...")
    # Adjust threshold to control edge density
    G = create_graph(co_occurrence, idx_to_keyphrase, threshold=0.05)
    
    print(Fore.GREEN + f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Output graph statistics
    output_graph_stats(G)
    
    # Visualize the graph
    viz_output_path = os.path.join(graph_output_path, 'keyphrase_graph_enhanced.png')
    print(Fore.GREEN + "Visualizing graph...")
    visualize_graph(G, output_path=viz_output_path)
    
    # Export graph data for GCN
    gcn_data_path = os.path.join(graph_output_path, 'gcn_graph_data_enhanced.json')
    print(Fore.GREEN + "Exporting graph data for GCN...")
    export_graph_for_gcn(G, gcn_data_path)
    
    print(Fore.RED + "Enhanced co-occurrence analysis complete!")

if __name__ == "__main__":
    main()