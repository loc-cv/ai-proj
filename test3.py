import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class MapExplorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Map Explorer")

        self.image_path = None
        self.image = None
        self.photo_image = None

        self.canvas = tk.Canvas(root)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.load_image_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_image_button.pack(side=tk.LEFT, padx=10)

        self.draw_lines_button = tk.Button(root, text="Draw Lines", command=self.draw_lines)
        self.draw_lines_button.pack(side=tk.RIGHT, padx=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            self.image_path = file_path
            self.load_and_display_image()

    def load_and_display_image(self):
        if self.image_path:
            self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.display_image()

    def display_image(self):
        if self.image is not None:
            self.photo_image = ImageTk.PhotoImage(Image.fromarray(self.image))
            self.canvas.config(width=self.image.shape[1], height=self.image.shape[0])
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

    def draw_lines(self):
        if self.image is not None:
            # Apply Canny edge detection
            edges = cv2.Canny(self.image, 50, 150)

            # Apply Hough Line Transform to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

            # Draw the lines on the original image
            line_image = np.zeros_like(self.image)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)

            # Display the image with lines
            self.image = cv2.addWeighted(self.image, 1, line_image, 1, 0)
            self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = MapExplorer(root)
    root.mainloop()
