import os
import sys
import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from matplotlib.patches import FancyArrowPatch

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description="Generate concept map from keyphrases")
parser.add_argument("--model", type=str, choices=["keybert", "ollama", "tfidf_ollama"], required=True)
parser.add_argument("--stemmed", action="store_true", help="Use stemmed keyphrases")
parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--threshold", type=float, default=0.4, help="Similarity threshold")
parser.add_argument("--edge_threshold", type=float, default=0.5, help="Cosine similarity for edge")
parser.add_argument("--export", action="store_true", help="Export graph to image file")
parser.add_argument("--keyword", type=str, help="Keyword for generating concept map (non-interactive mode)")
parser.add_argument("--max_nodes", type=int, default=25, help="Maximum number of nodes to include")
parser.add_argument("--max_depth", type=int, default=2, help="Maximum depth from root node")
parser.add_argument("--show_labels", action="store_true", help="Show relationship labels on edges")
args = parser.parse_args()

# -------------------------------
# Resolve Keyphrase Path
# -------------------------------
folder_prefix = os.path.join("data", "keyphrases", "sanitized", "stemmed" if args.stemmed else "")
key_phrases_path = os.path.join(folder_prefix, args.model)

if not os.path.exists(key_phrases_path):
    print(f"Sanitized keyphrases not found for {args.model}. Run preprocessing.")
    sys.exit(1)

print(f"Using sanitized keyphrases from: {key_phrases_path}")

# -------------------------------
# Load Keyphrases
# -------------------------------
all_keyphrases = set()
for file in os.listdir(key_phrases_path):
    if file.endswith(".txt"):
        with open(os.path.join(key_phrases_path, file), "r", encoding="utf-8") as f:
            for line in f:
                phrase = line.strip().lower()
                if phrase:
                    all_keyphrases.add(phrase)
all_keyphrases = list(all_keyphrases)
print(f"Total unique keyphrases loaded: {len(all_keyphrases)}")

# -------------------------------
# Embedding
# -------------------------------
try:
    embed_model = SentenceTransformer(args.embedding_model)
except:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(all_keyphrases, show_progress_bar=True)
print(f"Embedding shape: {embeddings.shape}")

# -------------------------------
# Get User Input or Use Command Line Args
# -------------------------------
if args.keyword:
    # Use command line argument if provided (non-interactive mode)
    keyword = args.keyword.strip().lower()
    max_nodes = args.max_nodes
    max_depth = args.max_depth
    print(f"Using keyword: '{keyword}' (max_nodes={max_nodes}, max_depth={max_depth})")
else:
    # Interactive mode
    keyword = input("Enter a keyword for generating the concept map: ").strip().lower()
    if not keyword:
        print("No keyword provided.")
        sys.exit(1)
        
    try:
        max_nodes = int(input("Max number of nodes to plot (e.g. 25): ").strip())
    except:
        max_nodes = 25
    try:
        max_depth = int(input("Max depth from keyword (e.g. 2): ").strip())
    except:
        max_depth = 2

# -------------------------------
# Similarity Matching
# -------------------------------
user_vec = embed_model.encode([keyword])[0]
sims = cosine_similarity(embeddings, user_vec.reshape(1, -1)).flatten()
selected_idxs = np.where(sims >= args.threshold)[0]
if len(selected_idxs) == 0:
    print(f"No matches found with threshold {args.threshold}. Trying with lower threshold...")
    # Try with a lower threshold if no matches found
    lower_threshold = args.threshold * 0.7  # 70% of the original threshold
    selected_idxs = np.where(sims >= lower_threshold)[0]
    if len(selected_idxs) == 0:
        print(f"Still no matches found with threshold {lower_threshold:.2f}.")
        print("Try a different keyword or check if the keyphrase extraction was successful.")
        sys.exit(1)
    else:
        print(f"Found {len(selected_idxs)} matches with lower threshold {lower_threshold:.2f}")
        args.threshold = lower_threshold
selected_phrases = [all_keyphrases[i] for i in selected_idxs]
selected_embeddings = embeddings[selected_idxs]

# Check if keyword is in the selected phrases
keyword_in_selected = keyword in selected_phrases

# If keyword is not in the selected phrases, add it
if not keyword_in_selected:
    print(f"Adding root keyword '{keyword}' to the set of phrases.")
    selected_phrases.append(keyword)
    keyword_embedding = user_vec
    selected_embeddings = np.vstack([selected_embeddings, keyword_embedding])

# Create similarity matrix with all selected phrases
sim_matrix = cosine_similarity(selected_embeddings)

print(f"Selected {len(selected_phrases)} keyphrases related to '{keyword}'")

# -------------------------------
# Relationship Types
# -------------------------------
# Define relationship type determiners based on semantic similarity patterns
def determine_relationship_type(source_phrase, target_phrase, similarity):
    """Determine the type of relationship between two phrases based on content and similarity score"""
    source_words = set(source_phrase.lower().split())
    target_words = set(target_phrase.lower().split())
    
    # Check for "is-a" relationships (hierarchical)
    if len(source_words) >= len(target_words) and target_words.issubset(source_words):
        return "is-a"  # Target is more specific version of source
    
    if len(target_words) >= len(source_words) and source_words.issubset(target_words):
        return "type-of"  # Source is more specific version of target
    
    # Check for "prerequisite-of" relationship based on teaching/learning terms
    prerequisite_terms = {"training", "learn", "model", "build", "data", "preprocessing", "feature"}
    if any(term in source_words for term in prerequisite_terms) and similarity > 0.6:
        return "prerequisite-of"
    
    # Common ML model and technique relationships
    ml_models = {"neural", "network", "svm", "tree", "forest", "regression", "cnn", "rnn", "transformer"}
    if (any(term in source_words for term in ml_models) and 
        any(term in target_words for term in ml_models) and
        similarity > 0.55):
        return "related-to"
    
    # Default to general relationship based on similarity strength
    if similarity > 0.7:
        return "strongly-related-to"
    return "related-to"

# -------------------------------
# Build Graph
# -------------------------------
try:
    root_idx = selected_phrases.index(keyword)
except ValueError:
    print(f"Critical error: Root keyword '{keyword}' still not found in selected set. Using first as fallback.")
    root_idx = 0

G = nx.DiGraph()
G.add_node(root_idx)
queue = deque([(root_idx, 0)])
visited = set([root_idx])

# First, try with original edge threshold
edge_threshold = args.edge_threshold
attempts = 0
max_attempts = 3

while attempts < max_attempts:
    queue = deque([(root_idx, 0)])
    visited = set([root_idx])
    G = nx.DiGraph()
    G.add_node(root_idx)
    
    while queue and len(G.nodes) < max_nodes:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        
        current_phrase = selected_phrases[current]
        for j in range(len(selected_phrases)):
            if j == current:
                continue
                
            similarity = sim_matrix[current][j]
            if similarity >= edge_threshold:
                target_phrase = selected_phrases[j]
                
                # Determine relationship type
                rel_type = determine_relationship_type(current_phrase, target_phrase, similarity)
                
                if j not in G:
                    G.add_node(j)
                    queue.append((j, depth + 1))
                    
                # Add edge with relationship type as an attribute
                G.add_edge(current, j, relationship=rel_type, weight=similarity)
    
    # Check if we have enough edges
    if G.number_of_edges() >= 2 or attempts == max_attempts - 1:
        break
    
    # If not enough edges, reduce threshold and try again
    attempts += 1
    edge_threshold *= 0.8  # Reduce threshold by 20%
    print(f"Not enough connections. Reducing edge threshold to {edge_threshold:.2f} (attempt {attempts}/{max_attempts})")
    
print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges with relationship types")

# -------------------------------
# GCN Embedding
# -------------------------------
refined = None
if G.number_of_edges() > 1:  # Need at least 2 edges for meaningful GCN
    try:
        node_map = {old: i for i, old in enumerate(G.nodes())}
        edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
        x = torch.tensor(np.array([selected_embeddings[i] for i in G.nodes()]), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        class GCN(torch.nn.Module):
            def __init__(self, in_c, hidden_c, out_c):
                super().__init__()
                self.conv1 = GCNConv(in_c, hidden_c)
                self.conv2 = GCNConv(hidden_c, out_c)
            def forward(self, d):
                x, edge_index = d.x, d.edge_index
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.conv2(x, edge_index)
                return x

        model_gcn = GCN(x.shape[1], 128, 64)
        refined = model_gcn(data)
        print(f"GCN embeddings shape: {refined.shape}")
    except Exception as e:
        print(f"Error in GCN processing: {e}")
        refined = None
else:
    print("Not enough edges for GCN. Simple graph will be displayed.")

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(14, 10))
ax = plt.gca()
pos = nx.spring_layout(G, k=0.6, seed=42)

# Compute depth for scaling
depth_map = {}
sim_map = {}
q = deque([(root_idx, 0)])
visited = set([root_idx])
while q:
    node, d = q.popleft()
    depth_map[node] = d
    sim_map[node] = sim_matrix[root_idx][node] if node != root_idx else 1.0
    for neighbor in G.successors(node):
        if neighbor not in visited:
            visited.add(neighbor)
            q.append((neighbor, d + 1))

# Normalize for color + size
max_depth_val = max(depth_map.values()) if depth_map and len(depth_map) > 0 else 1
# Ensure max_depth_val is at least 1 to avoid division by zero
max_depth_val = max(max_depth_val, 1)
node_sizes = [1200 / (depth_map.get(n, 1) + 1) * 2 for n in G.nodes()]
node_colors = [plt.cm.viridis(depth_map.get(n, 0) / max_depth_val) for n in G.nodes()]

# Draw nodes + labels
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors)
nx.draw_networkx_labels(
    G, pos, ax=ax,
    labels={n: selected_phrases[n] for n in G.nodes()},
    font_size=9,
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
)

# Define colors and styles for different relationship types
relationship_styles = {
    'is-a': {'color': 'darkblue', 'linestyle': '-', 'linewidth': 2},
    'type-of': {'color': 'navy', 'linestyle': '-', 'linewidth': 1.5},
    'prerequisite-of': {'color': 'darkred', 'linestyle': '-', 'linewidth': 2},
    'strongly-related-to': {'color': 'darkgreen', 'linestyle': '-', 'linewidth': 1.5},
    'related-to': {'color': 'gray', 'linestyle': '-', 'linewidth': 1}
}

# Draw straight arrows clipped to node border with relationship types
if G.number_of_edges() > 0:
    # Create a legend handlers and labels
    legend_elements = [
        plt.Line2D([0], [0], color=style['color'], lw=style['linewidth'], 
                   label=rel_type) 
        for rel_type, style in relationship_styles.items()
    ]
    
    # Add relationship labels to edges
    for u, v, data in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
            
        # Get relationship type and style
        rel_type = data.get('relationship', 'related-to')
        style = relationship_styles.get(rel_type, relationship_styles['related-to'])
        
        shrink = 0.07 * dist
        x_start = x1 + dx * shrink / dist
        y_start = y1 + dy * shrink / dist
        x_end = x2 - dx * shrink / dist
        y_end = y2 - dy * shrink / dist
        
        # Create arrow with relationship style
        arrow = FancyArrowPatch(
            (x_start, y_start), (x_end, y_end),
            arrowstyle='->', color=style['color'],
            linestyle=style['linestyle'],
            linewidth=style['linewidth'],
            mutation_scale=12
        )
        ax.add_patch(arrow)
        
        # Add relationship label near the middle of the edge
        if args.show_labels:
            x_mid = (x_start + x_end) / 2
            y_mid = (y_start + y_end) / 2
            # Small offset to prevent overlap with the edge
            offset = 0.05
            plt.text(x_mid + offset, y_mid + offset, rel_type, 
                     fontsize=7, alpha=0.7, 
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1))
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Colorbar legend
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_depth_val))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label("Depth from Root")

# Finish
rel_type_str = "with Typed Relationships" if G.number_of_edges() > 0 else ""
plt.title(f"Concept Map for '{keyword}' {rel_type_str}\n(depth={max_depth}, nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")
plt.axis("off")
plt.tight_layout()

# Create output directory if it doesn't exist
output_dir = os.path.join("data", "concept_maps")
os.makedirs(output_dir, exist_ok=True)

# Save the figure instead of showing it (to avoid backend issues)
filename = f"concept_map_{keyword.replace(' ', '_')}_{args.model}"
if args.stemmed:
    filename += "_stemmed"
output_path = os.path.join(output_dir, f"{filename}.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Concept map saved to: {output_path}")

# Try to show the plot, but don't fail if it doesn't work
try:
    plt.show()
except Exception as e:
    print(f"Note: Could not display plot interactively ({str(e)})")
    print("The image has been saved and can be viewed in the output directory.")
