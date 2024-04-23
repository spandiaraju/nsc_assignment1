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
    
    nc=100
    
    pca_full = PCA(n_components=nc)
    pca_full.fit(pixels_by_time_matrix.T)
    
    scores_full = pca_full.transform(pixels_by_time_matrix.T)
    
    reconstruction_errors = []
    
    for n_components in range(1, nc):
        pca_n = PCA(n_components=n_components)
    
        pca_n.components_ = pca_full.components_[:n_components]
        pca_n.mean_ = pca_full.mean_
        pca_n.explained_variance_ = pca_full.explained_variance_[:n_components]
        pca_n.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_components]
    
        scores_n = scores_full[:, :n_components]
        reconstructed_n = pca_n.inverse_transform(scores_n)
    
        error_n = np.mean((pixels_by_time_matrix.T - reconstructed_n) ** 2)
        reconstruction_errors.append(error_n)
    
    n_frames, height, width = frames.shape
    pixels_by_time_matrix = frames.reshape((n_frames, height * width)).T
    
    nc = 15
    
    reconstruction_errors_nmf = []
    for n_components in range(1, nc):
        model_nmf = NMF(n_components=n_components, init='random', random_state=0)
        W = model_nmf.fit_transform(pixels_by_time_matrix.T)
        H = model_nmf.components_
    
        reconstructed = np.dot(W, H)
    
        error = np.mean((pixels_by_time_matrix.T - reconstructed) ** 2)
        reconstruction_errors_nmf.append(error)

    plt.plot(reconstruction_errors, label = 'PCA')
    plt.plot(reconstruction_errors_nmf, label = 'NMF')
    
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs Number of Components')
    plt.grid(True)
    plt.xlim(0, 13)
    plt.ylim(9000, 13000)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()