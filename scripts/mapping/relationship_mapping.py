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
parser.add_argument("--model", type=str, choices=["keybert", "ollama"], required=True)
parser.add_argument("--stemmed", action="store_true", help="Use stemmed keyphrases")
parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")
parser.add_argument("--edge_threshold", type=float, default=0.7, help="Cosine similarity for edge")
parser.add_argument("--export", action="store_true", help="Export graph to image file")  # ✅ Add this line
args = parser.parse_args()

# -------------------------------
# Resolve Keyphrase Path
# -------------------------------
folder_prefix = os.path.join("data", "keyphrases", "sanitized", "stemmed" if args.stemmed else "")
key_phrases_path = os.path.join(folder_prefix, args.model)

if not os.path.exists(key_phrases_path):
    print(f"❌ Sanitized keyphrases not found for {args.model}. Run preprocessing.")
    sys.exit()

print(f"📁 Using sanitized keyphrases from: {key_phrases_path}")

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
print(f"📘 Total unique keyphrases loaded: {len(all_keyphrases)}")

# -------------------------------
# Embedding
# -------------------------------
try:
    embed_model = SentenceTransformer(args.embedding_model)
except:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(all_keyphrases, show_progress_bar=True)
print(f"✅ Embedding shape: {embeddings.shape}")

# -------------------------------
# Get User Input
# -------------------------------
keyword = input("Enter a keyword for generating the concept map: ").strip().lower()
if not keyword:
    print("❌ No keyword provided.")
    sys.exit()

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
    print("❌ No matches found.")
    sys.exit()
selected_phrases = [all_keyphrases[i] for i in selected_idxs]
selected_embeddings = embeddings[selected_idxs]
sim_matrix = cosine_similarity(selected_embeddings)

print(f"✅ Selected {len(selected_phrases)} keyphrases related to '{keyword}'")

# -------------------------------
# Build Graph
# -------------------------------
try:
    root_idx = selected_phrases.index(keyword)
except ValueError:
    print(f"⚠️ Root keyword '{keyword}' not found in selected set. Using first as fallback.")
    root_idx = 0

G = nx.DiGraph()
G.add_node(root_idx)
queue = deque([(root_idx, 0)])
visited = set([root_idx])

while queue and len(G.nodes) < max_nodes:
    current, depth = queue.popleft()
    if depth >= max_depth:
        continue
    neighbors = [
        j for j in range(len(selected_phrases))
        if sim_matrix[current][j] >= args.edge_threshold and j != current
    ]
    for nbr in neighbors:
        if nbr not in G:
            G.add_node(nbr)
            queue.append((nbr, depth + 1))
        G.add_edge(current, nbr)

print(f"📊 Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# -------------------------------
# GCN Embedding
# -------------------------------
if G.number_of_edges() > 0:
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
    print(f"🧠 GCN embeddings shape: {refined.shape}")
else:
    print("⚠️ Not enough edges for GCN.")

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
max_depth_val = max(depth_map.values()) if depth_map else 1
node_sizes = [1200 / (depth_map.get(n, 1) + 1) * 2 for n in G.nodes()]
node_colors = [plt.cm.viridis(depth_map[n] / max_depth_val) for n in G.nodes()]

# Draw nodes + labels
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors)
nx.draw_networkx_labels(
    G, pos, ax=ax,
    labels={n: selected_phrases[n] for n in G.nodes()},
    font_size=9,
    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
)

# Draw straight arrows clipped to node border
for u, v in G.edges():
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        continue
    shrink = 0.07 * dist
    x_start = x1 + dx * shrink / dist
    y_start = y1 + dy * shrink / dist
    x_end = x2 - dx * shrink / dist
    y_end = y2 - dy * shrink / dist
    arrow = FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle='->', color='gray',
        mutation_scale=12, linewidth=1
    )
    ax.add_patch(arrow)

# Colorbar legend
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_depth_val))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label("Depth from Root")

# Finish
plt.title(f"Concept Map for '{keyword}' (depth={max_depth}, nodes={G.number_of_nodes()})")
plt.axis("off")
plt.tight_layout()
plt.show()
