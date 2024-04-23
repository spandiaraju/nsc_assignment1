import plotly.graph_objs as go
from PIL import Image
import imageio
import numpy as np
import base64
import io
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from ipywidgets import interact, IntSlider
from skimage import io, filters, measure
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib.widgets import Slider
import matplotlib.animation as animation
from IPython.display import display, clear_output
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
from scipy.stats import pearsonr


tif_path = r'data/TEST_MOVIE_00001-small-motion.tif'
frames_rgb = imageio.v2.imread(tif_path)

def compute_correlation(A, B, dx, dy):
    A_x1, A_x2 = max(0, dx), min(A.shape[1], B.shape[1] + dx)
    A_y1, A_y2 = max(0, dy), min(A.shape[0], B.shape[0] + dy)

    B_x1, B_x2 = max(0, -dx), min(B.shape[1], A.shape[1] - dx)
    B_y1, B_y2 = max(0, -dy), min(B.shape[0], A.shape[0] - dy)

    A_cropped = A[A_y1:A_y2, A_x1:A_x2]
    B_cropped = B[B_y1:B_y2, B_x1:B_x2]

    if A_cropped.size == 0 or B_cropped.size == 0:
        return 0
    correlation, _ = pearsonr(A_cropped.flat, B_cropped.flat)
    return correlation

def correlation_heatmap(A, B, max_shift=10):
    """Generate a heatmap of correlations for shifts in range [-max_shift, max_shift] and print max correlation coordinates."""
    heatmap = np.zeros((2*max_shift+1, 2*max_shift+1))
    for dy in range(-max_shift, max_shift+1):
        for dx in range(-max_shift, max_shift+1):
            heatmap[dy + max_shift, dx + max_shift] = compute_correlation(A, B, dx, dy)

    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    max_dy, max_dx = max_idx[0] - max_shift, max_idx[1] - max_shift

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, origin='lower', interpolation='none', extent=[-max_shift, max_shift, -max_shift, max_shift])
    plt.colorbar()
    plt.title('Correlation Heatmap')
    plt.xlabel('Horizontal Shift (dx)')
    plt.ylabel('Vertical Shift (dy)')
    plt.grid(True)
    plt.scatter(max_dx, max_dy, color='red')
    plt.show()

    print(f"Maximum correlation at (dx, dy): ({max_dx}, {max_dy}) with correlation coefficient of {heatmap[max_idx]}")


def main():
    print("Selected frames number 204 and 209")
    A = frames_rgb[204]
    B = frames_rgb[209]
    heatmap = correlation_heatmap(A, B, max_shift=10)

if __name__ == '__main__':
    main()
