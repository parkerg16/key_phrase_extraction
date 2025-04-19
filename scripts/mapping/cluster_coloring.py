import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # pip install python-louvain

def assign_community_colors(G):
    try:
        partition = community_louvain.best_partition(G.to_undirected())
    except ImportError:
        print("⚠️ 'python-louvain' not installed. Falling back to greedy modularity.")
        communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
        partition = {}
        for i, group in enumerate(communities):
            for node in group:
                partition[node] = i

    return partition  # dict: node → community ID

def get_color_list(partition, colormap=plt.cm.Set3):
    unique_ids = sorted(set(partition.values()))
    color_map = {cid: colormap(i / len(unique_ids)) for i, cid in enumerate(unique_ids)}
    return [color_map[partition[n]] for n in partition]
