import cv2
import numpy as np

# Read the image
image = cv2.imread('./images/phohue-map.png', cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image, 50, 150)

# Define the parameters for Shi-Tomasi corner detection
max_corners = 999999999  # Increase this value to detect more corners
quality_level = 0.001
min_distance = 1

# Use Shi-Tomasi corner detection to find corners in the edges
corners = cv2.goodFeaturesToTrack(edges, max_corners, quality_level, min_distance)

# Convert corners to integers
corners = np.int0(corners)
print(len(corners))

# # Build a graph
# G = nx.Graph()

# # Add nodes to the graph
# for i, corner in enumerate(corners):
#     x, y = corner.ravel()
#     G.add_node(i, pos=(x, y))

# # Add edges to the graph based on proximity
# num_nodes = len(corners)
# for i in range(num_nodes):
#     for j in range(i + 1, num_nodes):
#         dist = np.linalg.norm(corners[i] - corners[j])
#         if dist < 30:  # Adjust this threshold based on your requirements
#             G.add_edge(i, j)

# # Draw the graph
# pos = nx.get_node_attributes(G, 'pos')
# nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=50, node_color='red', font_color='black')

# Display the result
# plt.show()

# Draw circles around the corners on the original image with red color
result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color image
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(result_image, (x, y), 2, (0, 0, 255), -1)  # (0, 0, 255) corresponds to red

# Display the result
cv2.imshow('Corners', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
