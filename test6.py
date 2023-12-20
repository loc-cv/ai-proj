import cv2
import numpy as np
import networkx as nx
import tkinter as tk
from PIL import Image, ImageTk
import json


class GraphBuilder:
    def __init__(self, master, image_path):
        self.master = master
        self.master.title("Graph Builder and Shortest Path Finder")

        self.image_path = image_path
        self.original_image = cv2.imread(self.image_path)
        self.G = nx.Graph()

        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.display_image()

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

        if len(self.G.nodes()) > 0:
            nearest_node = self.find_nearest_node(clicked_node)
            self.G.add_node(clicked_node)

            if nearest_node is not None:
                self.G.add_edge(clicked_node, nearest_node)
        else:
            # For the first click, just add the node without attempting to find its nearest node
            self.G.add_node(clicked_node)

        self.visualize_graph_on_image()
        self.save_graph_to_file()

    def find_nearest_node(self, target_node):
        _, nearest_node = min(
            (np.linalg.norm(np.array(target_node) - np.array(graph_node)), graph_node)
            for graph_node in self.G.nodes()
        )
        return nearest_node

    def visualize_graph_on_image(self):
        image_copy = self.original_image.copy()

        for edge in self.G.edges():
            node1 = tuple(map(int, edge[0]))
            node2 = tuple(map(int, edge[1]))
            cv2.line(image_copy, node1, node2, (0, 255, 0), 2)

        self.display_image(image_copy)

    def save_graph_to_file(self, filename="graph_data.json"):
        graph_data = {
            "nodes": list(self.G.nodes()),
            "edges": list(self.G.edges())
        }

        with open(filename, "w") as file:
            json.dump(graph_data, file)

    def load_graph_from_file(self, filename="graph_data.json"):
        try:
            with open(filename, "r") as file:
                graph_data = json.load(file)

            self.G.add_nodes_from(graph_data["nodes"])
            self.G.add_edges_from(graph_data["edges"])
            self.visualize_graph_on_image()

        except (FileNotFoundError, json.JSONDecodeError):
            print("Error loading graph data from file.")


def main():
    root = tk.Tk()
    image_path = "./images/phohue-map.png"
    app = GraphBuilder(root, image_path)

    # Load graph from file if it exists
    app.load_graph_from_file()

    root.mainloop()


if __name__ == "__main__":
    main()
