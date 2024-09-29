import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch


# helper function to draw the graph from the adjacency matrix
def plot_graph(adj_matrix, node_feats=None, remove_self_loops=True, device="cpu"):
    """
    Visualize a graph from an adjacency matrix.
    
    Parameters:
    adj_matrix (torch.Tensor): Adjacency matrix of shape [batch_size, num_nodes, num_nodes]
    remove_self_loops (bool): If True, removes the identity matrix from adj_matrix
    
    Returns:
    matplotlib.figure.Figure: The figure containing the graph visualization
    matplotlib.axes._subplots.AxesSubplot: The axes containing the graph visualization
    """
    # Ensure the input is a 3D tensor
    if adj_matrix.dim() == 2:
        adj_matrix = adj_matrix.unsqueeze(0).to(device)
    
    # Remove self-loops if specified
    if remove_self_loops:
        identity = torch.eye(adj_matrix.size(1)).unsqueeze(0).to(device)
        adj_matrix = adj_matrix - identity
    
    # Convert to numpy for NetworkX
    adj_np = adj_matrix.squeeze().to(device).numpy()

    # Create a graph
    G = nx.Graph(adj_np)

    # Set up the plot
    fig, ax = plt.subplots()

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    
    # Add edge labels
    edge_labels = {(u, v): '' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    # Add node features if provided
    if node_feats is not None:
        node_feats = node_feats.squeeze().cpu().numpy()
        for node, (x, y) in pos.items():
            features = node_feats[node]
            feature_str = ', '.join(f'{f:.2f}' for f in features)
            ax.text(x, y + 0.1, f'[{feature_str}]', ha='center', va='center')

    # Remove axis
    ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax
