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
import pickle

tif_path = r'data/TEST_MOVIE_00001-small.tif'
frames = imageio.v2.imread(tif_path)

def main():
    n_frames, height, width = frames.shape
    pixels_by_time_matrix = frames.reshape((n_frames, height * width)).T
    
    n_components_ica = 20  
    
    ica = FastICA(n_components=n_components_ica, random_state=0)
    ica.fit(pixels_by_time_matrix.T)  
    
    ica_components = ica.components_.reshape((n_components_ica, height, width))
    
    fig, axes = plt.subplots(5, 4, figsize=(15, 18))  
    for i, ax in enumerate(axes.flat):
        if i < n_components_ica:
            ax.imshow(ica_components[i], cmap='viridis') 
            ax.set_title(f'ICA Component {i+1}')
            ax.axis('off')
        else:
            ax.axis('off') 

plt.tight_layout()
plt.show()


if __name__ == '__main__':
    main()