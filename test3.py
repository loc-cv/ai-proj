import cv2
import numpy as np
import networkx as nx
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from scipy.spatial import Delaunay


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

        # Dilate the edges to ensure better connectivity
        edges = cv2.dilate(edges, kernel=np.ones((5, 5), dtype=np.uint8))

        # Find corners using Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(
            edges, maxCorners=100, qualityLevel=0.01, minDistance=10
        )

        # Create a graph
        G = nx.Graph()

        # Add corners as nodes to the graph
        for corner in corners.squeeze().astype(int):
            G.add_node(tuple(corner))

        # Perform Delaunay triangulation and add edges to the graph
        tri = Delaunay(corners.squeeze())
        edges_tri = tri.simplices
        for edge_tri in edges_tri:
            for i in range(3):
                corner1 = tuple(corners[edge_tri[i]].squeeze())
                corner2 = tuple(corners[edge_tri[(i + 1) % 3]].squeeze())
                G.add_edge(corner1, corner2)

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
    image_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")],
    )
    app = GraphBuilder(root, image_path)
    root.mainloop()


if __name__ == "__main__":
    main()
