import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from colorama import Fore, init
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

init(autoreset=True)

# Check for available hardware
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M-series GPU via MPS")
else:
    device = torch.device("cpu")
    print("Using CPU for computation")

# Paths
graph_data_path = 'graph_data'
gcn_output_path = 'gcn_output'

# Create output directory if it doesn't exist
if not os.path.exists(gcn_output_path):
    os.makedirs(gcn_output_path)
    print(f"Created folder: {gcn_output_path}")

class GCN(nn.Module):
    """Graph Convolutional Network model for keyphrase embeddings"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, edge_weight=None):
        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final GCN layer
        x = self.conv3(x, edge_index, edge_weight)
        
        return x

def load_graph_data(data_file):
    """Load graph data for GCN processing"""
    with open(data_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    return graph_data

def prepare_pyg_data(graph_data):
    """Prepare PyTorch Geometric data object from graph data"""
    # Extract data
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    edge_weights = graph_data['edge_weights']
    node_features = graph_data['node_features']
    
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    
    return data, nodes

def train_gcn(data, epochs=200):
    """Train the GCN model"""
    # Move data to device
    data = data.to(device)
    
    # Initialize model
    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = 32  # Dimension of final embeddings
    
    model = GCN(in_channels, hidden_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # Compute loss (using a contrastive loss for connected nodes)
        loss = compute_contrastive_loss(out, data.edge_index, data.edge_attr)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def compute_contrastive_loss(embeddings, edge_index, edge_weight):
    """Compute contrastive loss for GCN training"""
    # Extract source and target nodes for each edge
    src, dst = edge_index
    
    # Get embeddings for source and target nodes
    src_embeddings = embeddings[src]
    dst_embeddings = embeddings[dst]
    
    # Compute similarity (dot product)
    similarity = torch.sum(src_embeddings * dst_embeddings, dim=1)
    
    # Scale by edge weights
    weighted_similarity = similarity * edge_weight
    
    # Negative sampling (random node pairs)
    num_nodes = embeddings.size(0)
    num_neg_samples = edge_index.size(1) * 2
    
    neg_src = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
    
    # Get embeddings for negative pairs
    neg_src_embeddings = embeddings[neg_src]
    neg_dst_embeddings = embeddings[neg_dst]
    
    # Compute negative similarity
    neg_similarity = torch.sum(neg_src_embeddings * neg_dst_embeddings, dim=1)
    
    # Compute loss (maximize similarity for connected nodes, minimize for random pairs)
    pos_loss = -torch.mean(weighted_similarity)
    neg_loss = torch.mean(F.relu(neg_similarity + 0.3))  # Margin of 0.3
    
    return pos_loss + neg_loss

def visualize_embeddings(embeddings, nodes, output_path=None):
    """Visualize node embeddings using t-SNE"""
    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Perform clustering to color-code nodes
    kmeans = KMeans(n_clusters=8, random_state=42)
    clusters = kmeans.fit_predict(reduced_embeddings)
    
    # Plotting
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                         c=clusters, cmap='viridis', alpha=0.7, s=50)
    
    # Add labels for some important nodes (top 20 by degree centrality)
    # This would be better if we had a centrality measure, but for simplicity
    # we'll just label a few random nodes
    num_to_label = min(20, len(nodes))
    indices_to_label = np.random.choice(len(nodes), num_to_label, replace=False)
    
    for idx in indices_to_label:
        plt.annotate(nodes[idx], (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]),
                    fontsize=8, alpha=0.8)
    
    plt.title("t-SNE Visualization of Keyphrase Embeddings")
    plt.colorbar(scatter, label="Cluster")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Embedding visualization saved to {output_path}")
    
    plt.close()

def export_embeddings(embeddings, nodes, output_path):
    """Export the generated embeddings for further use"""
    # Create a dictionary mapping phrases to their embeddings
    embeddings_dict = {
        nodes[i]: embeddings[i].tolist() for i in range(len(nodes))
    }
    
    # Export to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_dict, f, indent=2)
    
    print(f"Embeddings exported to {output_path}")

def main():
    # Load graph data
    data_file = os.path.join(graph_data_path, 'gcn_graph_data.json')
    print(Fore.GREEN + f"Loading graph data from {data_file}...")
    graph_data = load_graph_data(data_file)
    
    # Prepare data for PyTorch Geometric
    print(Fore.GREEN + "Preparing data for GCN...")
    pyg_data, nodes = prepare_pyg_data(graph_data)
    
    # Train GCN model
    print(Fore.GREEN + "Training GCN model...")
    model = train_gcn(pyg_data)
    
    # Generate embeddings
    print(Fore.GREEN + "Generating keyphrase embeddings...")
    model.eval()
    with torch.no_grad():
        embeddings = model(pyg_data.x.to(device), 
                          pyg_data.edge_index.to(device), 
                          pyg_data.edge_attr.to(device))
        embeddings = embeddings.cpu().numpy()
    
    # Visualize embeddings
    viz_output_path = os.path.join(gcn_output_path, 'keyphrase_embeddings.png')
    print(Fore.GREEN + "Visualizing embeddings...")
    visualize_embeddings(embeddings, nodes, output_path=viz_output_path)
    
    # Export embeddings
    embeddings_path = os.path.join(gcn_output_path, 'keyphrase_embeddings.json')
    print(Fore.GREEN + "Exporting embeddings...")
    export_embeddings(embeddings, nodes, embeddings_path)
    
    print(Fore.RED + "GCN processing complete!")

if __name__ == "__main__":
    main()