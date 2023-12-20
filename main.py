import cv2
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("./images/phohue-map.png", cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, 50, 150)

# Find coordinates of edge pixels
edge_coordinates = np.argwhere(edges > 0)

# Create a KD-tree for efficient neighbor search
tree = cKDTree(edge_coordinates)

# Create a graph
G = nx.Graph()

# Connect nearby edge pixels to form edges in the graph
for i in range(len(edge_coordinates)):
    node1 = tuple(edge_coordinates[i])

    # Find nearby nodes within a distance threshold
    nearby_nodes = tree.query_ball_point(node1, r=10)  # Adjust the radius as needed

    for j in nearby_nodes:
        if i < j:  # Ensure not to add duplicate edges
            node2 = tuple(edge_coordinates[j])
            G.add_edge(node1, node2)

# Visualize the graph
pos = {node: node for node in G.nodes()}
nx.draw(G, pos, node_size=5, with_labels=False)
plt.show()
