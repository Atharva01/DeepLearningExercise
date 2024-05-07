import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Checker:
    def __init__(self, resolution, tile_size) -> None:
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        num_tile = self.resolution // self.tile_size
        white_square = np.ones((self.tile_size, self.tile_size))
        black_square = np.zeros((self.tile_size, self.tile_size))
        row_black = np.concatenate(
            [black_square, white_square] * (num_tile // 2), axis=1)
        row_white = np.concatenate(
            [white_square, black_square] * (num_tile // 2), axis=1)
        checker = np.concatenate(
            [row_black, row_white] * (num_tile // 2), axis=0)
        self.output = np.array(checker)
        return self.output.copy()

    def show(self):
        # print(f"{type(self.checker)}")
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    def __init__(self, resolution=1000, radius=200, position=(500, 250)):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((resolution, resolution), dtype=int)
        self.draw()

    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        xx, yy = np.meshgrid(x, y)

        # Formula of a circle: (x - x0)^2 + (y - y0)^2 = r^2
        circle = (xx - self.position[0])**2 + \
            (yy - self.position[1])**2 <= self.radius**2
        self.output[circle] = 1
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum:
    def __init__(self, resolution) -> None:
        self.resolution = resolution

    def draw(self):
        top_left = np.array([0, 0, 255])  # Blue
        top_right = np.array([255, 0, 0])  # Red
        bottom_right = np.array([255, 255, 0])  # Yellow
        bottom_left = np.array([0, 255, 255])  # Turquoise
        # corner_color = {"top_left": top_left, "top_right": top_right,
        #                "bottom_left": bottom_left, "bottom_right": bottom_right}
        # Create a horizontal linear interpolation for each vertical gradient
        top_left, top_right, bottom_left, bottom_right = [
            x / 255.0 for x in [top_left, top_right, bottom_left, bottom_right]]
        # Create a vertical linear interpolation between the top and bottom colors
        left_gradient = np.linspace(
            bottom_left, top_left, self.resolution)[:, None]
        right_gradient = np.linspace(
            bottom_right, top_right, self.resolution)[:, None]
        gradient = np.linspace(left_gradient, right_gradient, self.resolution)[
            :, :, None]
        gradient = gradient.reshape(self.resolution, self.resolution, 3)
        self.output = np.rot90(gradient)
        return self.output.copy()

    def show(self) -> None:
        plt.imshow(self.output)
        return None


if __name__ == "__main__()":
    pass
