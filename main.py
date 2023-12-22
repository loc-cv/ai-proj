import cv2
import numpy as np
import networkx as nx
import tkinter as tk
from PIL import Image, ImageTk
import json
from scipy.spatial import cKDTree
import heapq


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

        self.start_node = None
        self.end_node = None
        self.shortest_path = None

        # Add a Reset button
        reset_button = tk.Button(self.master, text="Reset", command=self.reset_program)
        reset_button.pack()

        self.display_image()

    def load_graph_from_file(self, filename="graph_data.json"):
        try:
            with open(filename, "r") as file:
                graph_data = json.load(file)

            nodes = [tuple(node) for node in graph_data["nodes"]]
            edges = [
                tuple((tuple(edge[0]), tuple(edge[1]))) for edge in graph_data["edges"]
            ]

            self.G.add_nodes_from(nodes)
            self.G.add_edges_from(edges)

            self.visualize_graph_on_image()

        except (FileNotFoundError, json.JSONDecodeError):
            print("Error loading graph data from file.")

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
        nearest_node = self.find_nearest_node(clicked_node)
        self.G.add_edge(clicked_node, nearest_node)
        self.G.add_node(clicked_node)
        if self.start_node is None:
            self.start_node = clicked_node
        elif self.end_node is None:
            self.end_node = clicked_node

        if self.start_node is not None and self.end_node is not None:
            self.calculate_and_display_shortest_path(
                start_node=self.start_node, end_node=self.end_node
            )

    def find_nearest_node(self, target_node):
        _, nearest_node = min(
            (np.linalg.norm(np.array(target_node) - np.array(graph_node)), graph_node)
            for graph_node in self.G.nodes()
            if target_node != graph_node
        )
        return nearest_node

    def calculate_and_display_shortest_path(self, start_node, end_node):
        try:
            shortest_path = self.a_star_search(start_node, end_node)
            self.shortest_path = shortest_path
            self.display_shortest_path()
        except ValueError:
            print("No path found between the selected nodes.")

    def a_star_search(self, start, end):
        open_set = []
        closed_set = set()

        # Each element in the open_set is a tuple (f_cost, node)
        heapq.heappush(open_set, (0, start))

        # Keep track of the cost from start to each node
        g_costs = {node: float("inf") for node in self.G.nodes()}
        g_costs[start] = 0

        # Keep track of the estimated total cost from start to end through each node
        f_costs = {node: float("inf") for node in self.G.nodes()}
        f_costs[start] = self.heuristic(start, end)

        # Keep track of the parent node for each node
        parents = {node: None for node in self.G.nodes()}

        while open_set:
            current_cost, current_node = heapq.heappop(open_set)

            if current_node == end:
                # Reconstruct the path
                path = [end]
                while parents[end] is not None:
                    path.append(parents[end])
                    end = parents[end]
                return path[::-1]

            closed_set.add(current_node)

            for neighbor in self.G.neighbors(current_node):
                if neighbor in closed_set:
                    continue

                tentative_g_cost = g_costs[current_node] + self.distance(current_node, neighbor)
                if tentative_g_cost < g_costs[neighbor]:
                    parents[neighbor] = current_node
                    g_costs[neighbor] = tentative_g_cost
                    f_costs[neighbor] = g_costs[neighbor] + self.heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_costs[neighbor], neighbor))

        raise ValueError("No path found")

    def heuristic(self, node, goal):
        # Euclidean distance heuristic
        return np.linalg.norm(np.array(node) - np.array(goal))

    def distance(self, node1, node2):
        # Weighted distance between nodes
        return np.linalg.norm(np.array(node1) - np.array(node2))


    def visualize_graph_on_image(self):
        image_copy = self.original_image.copy()

        for edge in self.G.edges():
            node1 = tuple(map(int, edge[0]))
            node2 = tuple(map(int, edge[1]))
            cv2.line(image_copy, node1, node2, (0, 255, 0), 2)

        self.display_image(image_copy)

    def display_shortest_path(self):
        # Highlight the shortest path on the image
        image_copy = self.original_image.copy()
        for i in range(len(self.shortest_path) - 1):
            node1 = tuple(map(int, self.shortest_path[i]))
            node2 = tuple(map(int, self.shortest_path[i + 1]))
            cv2.line(image_copy, node1, node2, (0, 0, 255), 2)

        # Update the displayed image with the highlighted path
        self.display_image(image_copy)

    def reset_program(self):
        self.start_node = None
        self.end_node = None
        self.shortest_path = None
        self.G = nx.Graph()
        self.graph_data_path = "./graph_data.json"  # Reset the graph data path
        self.canvas.delete("all")  # Clear the canvas
        self.display_image()

        # Load the graph from the file after resetting
        self.load_graph_from_file(self.graph_data_path)

def main():
    root = tk.Tk()
    image_path = "./images/phohue-map.png"
    graph_data_path = "./graph_data.json"
    app = GraphBuilder(root, image_path)

    app.load_graph_from_file()

    root.mainloop()


if __name__ == "__main__":
    main()
