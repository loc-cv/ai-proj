import cv2
import numpy as np
import networkx as nx
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from scipy.spatial import cKDTree


class GraphBuilder:
    def __init__(self, master, image_path):
        self.master = master
        self.master.title("Graph Builder and Shortest Path Finder")

        self.image_path = image_path
        self.original_image = cv2.imread(self.image_path)
        self.G = self.build_graph()

        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.start_node = None
        self.end_node = None
        self.shortest_path = None

        self.display_image()

    def build_graph(self):
        # Apply Canny edge detection
        edges = cv2.Canny(self.original_image, 50, 150)

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
            nearby_nodes = tree.query_ball_point(
                node1, r=10
            )  # Adjust the radius as needed

            for j in nearby_nodes:
                if i < j:  # Ensure not to add duplicate edges
                    node2 = tuple(edge_coordinates[j])
                    distance = np.linalg.norm(np.array(node1) - np.array(node2))

                    # You can adjust the threshold for connecting nodes based on distance
                    if distance < 5:  # You may need to experiment with this threshold
                        G.add_edge(node1, node2)

        return G

    def display_image(self, image=None):
        if image is None:
            image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=photo.width(), height=photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def on_canvas_click(self, event):
        clicked_node = (int(event.x), int(event.y))

        if self.start_node is None:
            # For the first click, find and store the nearest node as the start node
            self.start_node = self.find_nearest_node(clicked_node)
            print("Start Node:", self.start_node)
            self.G.add_node(self.start_node)
        elif self.end_node is None:
            # For the second click, find and store the nearest node as the end node
            self.end_node = self.find_nearest_node(clicked_node)
            print("End Node:", self.end_node)
            self.G.add_node(self.end_node)
            self.find_and_display_shortest_path()

    def find_nearest_node(self, target_node):
        # Find the nearest node in the graph to the target node
        _, nearest_node = min(
            (np.linalg.norm(np.array(target_node) - np.array(graph_node)), graph_node)
            for graph_node in self.G.nodes()
        )

        return nearest_node

    def find_and_display_shortest_path(self):
        if self.start_node is not None and self.end_node is not None:
            try:
                self.shortest_path = nx.shortest_path(
                    self.G, source=self.start_node, target=self.end_node
                )
                self.display_shortest_path()
            except nx.NetworkXNoPath:
                print(
                    "No direct path found between the selected nodes. Attempting to find a path through the graph."
                )
                self.find_and_display_path_through_graph()

    def find_and_display_path_through_graph(self):
        # Find a path through the graph
        try:
            self.shortest_path = nx.shortest_path(self.G, source=self.start_node)
            self.shortest_path += nx.shortest_path(self.G, target=self.end_node)[1:]
            self.display_shortest_path()
        except nx.NetworkXNoPath:
            print("No path found through the graph between the selected nodes.")
            self.clear_selected_nodes()

    def display_shortest_path(self):
        # Highlight the shortest path on the image
        image_copy = self.original_image.copy()
        for i in range(len(self.shortest_path) - 1):
            node1 = tuple(map(int, self.shortest_path[i]))
            node2 = tuple(map(int, self.shortest_path[i + 1]))
            cv2.line(image_copy, node1, node2, (0, 255, 0), 2)

        # Update the displayed image with the highlighted path
        self.display_image(image_copy)

    def clear_selected_nodes(self):
        # Clear the selected start and end nodes
        self.start_node = None
        self.end_node = None


def main():
    root = tk.Tk()
    image_path = "./images/phohue-map.png"
    app = GraphBuilder(root, image_path)
    root.mainloop()


if __name__ == "__main__":
    main()
