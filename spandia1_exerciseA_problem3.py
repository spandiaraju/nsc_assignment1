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
    frames_variance = np.var(frames, axis=0)
    summary_image = frames_variance
    
    filtered_image = filters.gaussian(summary_image, sigma=2.5)
    
    thresh = filters.threshold_otsu(filtered_image)
    binary_mask = filtered_image > thresh
    
    seed_pixel = (350, 50)
        

    labeled_image = measure.label(binary_mask)
    roi_mask = labeled_image == labeled_image[seed_pixel]
    
    plt.imshow(summary_image)
    plt.imshow(roi_mask, alpha=0.40, cmap = 'ocean') 
    plt.show()

    labeled_mask = measure.label(binary_mask)
    
    regions = measure.regionprops(labeled_mask)
    
    plt.imshow(summary_image)
    
    for region in regions:
        y, x = region.centroid
        plt.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10, markeredgewidth=2)
    
        print(f'Region label {region.label}: Centroid at (x={x}, y={y})')
    
    plt.show()

    regions = measure.regionprops(labeled_mask)
    
    roi_masks = []

    i = 0

    for region in regions:
        mask = labeled_mask == region.label
        roi_masks.append(mask)
    
        plt.figure()
        plt.imshow(summary_image, cmap='copper')
        plt.imshow(mask, alpha=0.5, cmap='ocean')
        if i == 4:
            plt.title(f'Excluded ROI {region.label}')
        else:
            plt.title(f'ROI {region.label}')
            
        plt.show()
        i += 1

    roi_masks = np.array(roi_masks)
    roi_masks = np.delete(roi_masks, 4, axis=0)

    
if __name__ == '__main__':
    main()