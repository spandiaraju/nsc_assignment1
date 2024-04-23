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

def calculate_correlation_with_neighbors(frames, x, y):
    time_series_center = frames[:, x, y]

    neighbors = []
    if x > 0:  # Left neighbor
        neighbors.append(frames[:, x-1, y])
    if x < frames.shape[2] - 1: 
        neighbors.append(frames[:, x+1, y])
    if y > 0:  # Up neighbor
        neighbors.append(frames[:, x, y-1])
    if y < frames.shape[1] - 1: 
        neighbors.append(frames[:, x, y+1])

    correlations = [pearsonr(time_series_center, n)[0] for n in neighbors]
    return np.nanmean(correlations)

def main():


    with open(r'correlation_image.pkl', 'rb') as f:
        correlation_image = pickle.load(f)

    frames_rgb = frames
    frames_mean = np.mean(frames_rgb, axis=0)
    frames_variance = np.var(frames_rgb, axis=0)
    frames_median = np.median(frames_rgb, axis=0)
    frames_max = np.max(frames_rgb, axis=0)
    frames_min = np.min(frames_rgb, axis=0)
    frames_diff = frames_max - frames_mean
    mean_pixel_values = np.mean(frames, axis=0)
    std_pixel_values = np.std(frames, axis=0)
    
    std_pixel_values[std_pixel_values == 0] = np.nan
    
    corr_var_values = std_pixel_values/mean_pixel_values
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))  
    axes = axes.ravel() 
    
    ax = axes[0]
    cax = ax.imshow(frames_max)
    ax.set_title('Maximum Image')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[1]
    cax = ax.imshow(frames_diff)
    ax.set_title('Diff Image (Max - Mean)')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[2]
    cax = ax.imshow(corr_var_values)
    ax.set_title('Coefficient of Variance Image')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    ax = axes[3]
    cax = ax.imshow(correlation_image)
    ax.set_title('Neigbour Correlation Image')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()