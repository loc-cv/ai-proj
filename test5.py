import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("./images/phohue.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to create a binary mask for the darkest areas
_, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Invert the mask (optional)
mask_inv = cv2.bitwise_not(mask)

# Apply Canny edge detection to the original image using the mask
edges = cv2.Canny(image, 100, 200)

# Apply the mask to the edges to keep only the edges in the darkest areas
edges_in_darkest = cv2.bitwise_and(edges, edges, mask=mask)

# # Display the results
# # cv2.imshow('Original Image', image)
# cv2.imshow('Canny Edges in Darkest Areas', edges_in_darkest)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Find contours in the edges
contours, _ = cv2.findContours(
    edges_in_darkest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Extract nodes from the contours
nodes = []
for contour in contours:
    for point in contour:
        x, y = point[0]
        nodes.append((x, y))

# Build a graph using NetworkX
G = nx.Graph()

# Add nodes to the graph
G.add_nodes_from(nodes)

# Add edges between nearby nodes
for node1 in nodes:
    for node2 in nodes:
        if node1 != node2:
            distance = np.linalg.norm(np.array(node1) - np.array(node2))
            if distance < 20:  # Adjust the threshold based on your requirements
                G.add_edge(node1, node2)

# Draw the graph
pos = {
    node: (node[0], -node[1]) for node in G.nodes()
}  # Invert y-axis for image-like plot
nx.draw(G, pos, with_labels=False, node_size=5)
plt.show()
