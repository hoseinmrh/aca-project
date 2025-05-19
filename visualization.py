import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def visualize_protein_graph(data):
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, dim=3, seed=0)
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection="3d")
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    ax.scatter(*node_xyz.T, s=500, c="#0A047A")
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")
    plt.show()
