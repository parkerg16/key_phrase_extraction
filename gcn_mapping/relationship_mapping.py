import torch
import torch.nn.functional as F
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -------------------------------
# Step 1: Aggregate Key Phrases
# -------------------------------
key_phrases_path = 'key_phrases'
all_keyphrases = set()

for file_name in os.listdir(key_phrases_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(key_phrases_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    phrase = line.split(':')[0].strip()
                    if phrase:
                        all_keyphrases.add(phrase)

all_keyphrases = list(all_keyphrases)
print(f"Total unique key phrases: {len(all_keyphrases)}")

# -------------------------------
# Step 2: Compute Embeddings for All Key Phrases
# -------------------------------
model = SentenceTransformer("fine_tuned_model")
embeddings = model.encode(all_keyphrases, show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)

# -------------------------------
# Step 3: Filter Key Phrases Based on User Keyword
# -------------------------------
user_keyword = input("Enter a keyword for generating the concept map: ").strip()
if not user_keyword:
    print("No keyword provided. Exiting.")
    exit()

# Compute embedding for the user-provided keyword
user_embedding = model.encode([user_keyword])[0]

# Calculate cosine similarity between the user keyword and each key phrase
similarities = cosine_similarity(embeddings, user_embedding.reshape(1, -1)).flatten()

# Set a threshold to select key phrases related to the user keyword (adjust as needed)
filter_threshold = 0.5
selected_indices = np.where(similarities >= filter_threshold)[0]

if len(selected_indices) == 0:
    print("No key phrases found with sufficient similarity to the provided keyword.")
    exit()

selected_keyphrases = [all_keyphrases[i] for i in selected_indices]
selected_embeddings = embeddings[selected_indices]
print(f"Selected {len(selected_keyphrases)} key phrases related to '{user_keyword}'.")

# -------------------------------
# Step 4: Build Graph Using Cosine Similarity on Filtered Key Phrases
# -------------------------------
similarity_matrix = cosine_similarity(selected_embeddings)
similarity_threshold = 0.7  # Adjust threshold for creating edges

edge_index = [[], []]
num_nodes = len(selected_keyphrases)
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j and similarity_matrix[i][j] >= similarity_threshold:
            edge_index[0].append(i)
            edge_index[1].append(j)

edge_index = torch.tensor(edge_index, dtype=torch.long)
x = torch.tensor(selected_embeddings, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# -------------------------------
# Step 5: Define and Run the GCN Model
# -------------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

in_channels = selected_embeddings.shape[1]
hidden_channels = 128  # Adjust as needed
out_channels = 64      # Final feature representation dimension

model_gcn = GCN(in_channels, hidden_channels, out_channels)
refined_embeddings = model_gcn(data)
print("Refined embeddings shape:", refined_embeddings.shape)

# -------------------------------
# Step 6: Visualize the Concept Map
# -------------------------------
# Create a NetworkX graph from the edge index
G = nx.Graph()
G.add_nodes_from(range(num_nodes))
for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
    G.add_edge(i, j)

# Create a mapping from node index to key phrase for labeling
labels = {i: selected_keyphrases[i] for i in range(num_nodes)}

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
nx.draw_networkx_edges(G, pos, width=1)
nx.draw_networkx_labels(G, pos, labels, font_size=10)
plt.title(f"Concept Map for '{user_keyword}'")
plt.axis('off')
plt.show()
