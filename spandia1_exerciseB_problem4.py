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

#py_file_location = "/content/drive/MyDrive/nsc 1"
#sys.path.append(os.path.abspath(py_file_location))

tif_path = r'data/TEST_MOVIE_00001-small.tif'
frames = imageio.v2.imread(tif_path)

def track_roi_fluorescence(frames, mask, type="mean"):

    time_series = []

    for frame in frames:
        roi_pixels = frame[mask]

        if type == "mean":
          average_fluorescence = roi_pixels.mean()
          time_series.append(average_fluorescence)
            
        elif type == "median":
          average_fluorescence = roi_pixels.median()
          time_series.append(average_fluorescence)

        elif type == "max":
          max_fluorescence = roi_pixels.max()
          time_series.append(max_fluorescence)

        elif type == "std":
          std_fluorescence = roi_pixels.std()
          time_series.append(std_fluorescence)

        else:
          raise ValueError("Invalid type. Choose 'mean', 'median', 'max', or 'std'.")

    return np.array(time_series)


def update_plot(frame_num, roi_index, ax1, ax2, im, line1, line2):
    masked_image = np.where(roi_masks[roi_index], frames_rgb[frame_num], 0)
    im.set_data(masked_image)

    line1.set_xdata([frame_num, frame_num])
    line2.set_xdata([frame_num, frame_num])

    return [im, line1, line2]

def play_sequence_with_trace(roi_masks, frames_rgb, roi_index, all_average_fluorescence_time_series, all_max_fluorescence_time_series):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    masked_image = np.where(roi_masks[roi_index], frames_rgb[0], 0)
    im = ax1.imshow(masked_image)
    ax1.axis('off')
    ax1.set_title('ROI Masked Frame')

    ax2.plot(all_average_fluorescence_time_series[roi_index], label='Average Fluorescence')
    ax2.plot(all_max_fluorescence_time_series[roi_index], label='Max Fluorescence')
    line1, = ax2.plot([0, 0], [np.min(all_average_fluorescence_time_series[roi_index]), np.max(all_max_fluorescence_time_series[roi_index])], 'r-')
    line2, = ax2.plot([0, 0], [np.min(all_average_fluorescence_time_series[roi_index]), np.max(all_max_fluorescence_time_series[roi_index])], 'r-')
    ax2.legend()
    ax2.set_xlabel('Time (frames)')
    ax2.set_ylabel('Fluorescence')
    ax2.set_title('Fluorescence Time Series')

    anim = FuncAnimation(fig, update_plot, frames=len(frames_rgb), fargs=(roi_index, ax1, ax2, im, line1, line2), blit=True, interval=100)
    plt.close(fig) 

    return HTML(anim.to_html5_video())





tif_path = r'data/TEST_MOVIE_00001-small.tif'
frames = imageio.v2.imread(tif_path)
frames_rgb = frames
frames_variance = np.var(frames, axis=0)
summary_image = frames_variance

filtered_image = filters.gaussian(summary_image, sigma=2.5)

thresh = filters.threshold_otsu(filtered_image)
binary_mask = filtered_image > thresh

seed_pixel = (350, 50) 

labeled_image = measure.label(binary_mask)
roi_mask = labeled_image == labeled_image[seed_pixel]

labeled_mask = measure.label(binary_mask)

regions = measure.regionprops(labeled_mask)

roi_masks = []
for region in regions:

    mask = labeled_mask == region.label
    roi_masks.append(mask)

roi_masks = np.array(roi_masks)
roi_masks = np.delete(roi_masks, 4, axis=0)
all_average_fluorescence_time_series = []
for i in range(len(roi_masks)):
    mask = roi_masks[i]
    fluorescence_time_series = track_roi_fluorescence(frames, mask, type="mean")
    all_average_fluorescence_time_series.append(fluorescence_time_series)

all_max_fluorescence_time_series = []
for i in range(len(roi_masks)):
    mask = roi_masks[i]
    fluorescence_time_series = track_roi_fluorescence(frames, mask, type="max")
    all_max_fluorescence_time_series.append(fluorescence_time_series)

all_std_fluorescence_time_series = []
for i in range(len(roi_masks)):
    mask = roi_masks[i]
    fluorescence_time_series = track_roi_fluorescence(frames, mask, type="std")
    all_std_fluorescence_time_series.append(fluorescence_time_series)

def main():
    for i in range(5):
        index = np.arange(len(all_average_fluorescence_time_series[i]))
        
        plt.plot(index, all_average_fluorescence_time_series[i], label = "Average Fluorescence")
        
        plt.fill_between(index,
                         all_average_fluorescence_time_series[i] - all_std_fluorescence_time_series[i],
                         all_average_fluorescence_time_series[i] + all_std_fluorescence_time_series[i],
                         color='blue', alpha=0.2)
        
        plt.xlabel('Time (frames)')
        plt.ylabel('Fluorescence')
        plt.title(f'Fluorescence Time Series of ROI {i+1}')
        plt.legend()
        plt.show()

    # To play the sequence with ffmpeg working:
    #play_sequence_with_trace(roi_masks, frames, 0, all_average_fluorescence_time_series, all_max_fluorescence_time_series)


if __name__ == '__main__':
    main()