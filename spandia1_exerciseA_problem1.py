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

def display_frame(frame_num):
          plt.figure(figsize=(10, 10))
          plt.imshow(frames_rgb[frame_num])
          plt.axis('off')
          plt.show()

def update(frame_num, frames_rgb):
    fig, ax = plt.subplots()
    im = ax.imshow(frames_rgb[0])
    im.set_data(frames_rgb[frame_num])
    plt.axis('off')
    plt.imshow(frames_rgb[frame_num])

def view_image_sequence(frames, play=True):
    if play:
        for frame in range(len(frames)):
            clear_output(wait=True)
            update(frame, frames_rgb)
            plt.pause(0.1)
    else:
        interact(display_frame, frame_num=IntSlider(min=0, max=499, step=1, value=0, description='Frame Number:'))

def main():
    view_image_sequence(frames_rgb, play=True)

if __name__ == '__main__':
    main()
