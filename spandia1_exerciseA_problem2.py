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


tif_path = r'data/TEST_MOVIE_00001-small.tif'
frames_rgb = imageio.v2.imread(tif_path)

def main():
    frames_mean = np.mean(frames_rgb, axis=0)
    frames_variance = np.var(frames_rgb, axis=0)
    frames_median = np.median(frames_rgb, axis=0)
    frames_max = np.max(frames_rgb, axis=0)
    frames_min = np.min(frames_rgb, axis=0)
    frames_diff = frames_max - frames_mean

    fig, axes = plt.subplots(2, 2, figsize=(12, 12)) 
    axes = axes.ravel()
    
    ax = axes[0]  
    cax = ax.imshow(frames_mean)
    ax.set_title('Mean Image')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1]
    cax = ax.imshow(frames_median)
    ax.set_title('Median Image')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[2] 
    cax = ax.imshow(frames_variance)
    ax.set_title('Variance Image')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    
    fig.delaxes(axes[3])
    
    plt.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    main()
    