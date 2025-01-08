import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.interpolate import griddata


def get_2d_array_peaks(image, min_distance=10, num_peaks=3, peak_radius=2):
    """Find the peaks in a 2D array and return the coordinates of the top num_peaks peaks."""

    neighborhood = filters.maximum_filter(image, size=min_distance)
    local_max = (image == neighborhood)
    peak_coords = np.argwhere(local_max)
    peak_intensities = image[local_max]
    sorted_indices = np.argsort(peak_intensities)[::-1]
    sorted_peak_coords = peak_coords[sorted_indices]

    return sorted_peak_coords[:num_peaks]


def kmean_cluster_2d_array(array, n_clusters=2):

    def morphological_operator(binary_image, size:tuple):
        kernel = np.ones(size, np.uint8)
        eroded_image = cv2.erode(binary_image, kernel, iterations=1)
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        return closed_image
    
    def get_cluster_sum(centers,image, cluster):
        # sorted_indices = np.argsort(centers)[::-1]
        # highest_value_clusters = sorted_indices[:2]
        return np.sum(image[np.isin(cluster, 1)])

    flattened_array = array.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flattened_array)
    clustered_array = kmeans.labels_.reshape(array.shape)
    cluster_centers = kmeans.cluster_centers_.flatten()
    sum_values1 = get_cluster_sum(cluster_centers, array, clustered_array)

    clustered_array = clustered_array.astype(array.dtype)
    closed_image = morphological_operator(clustered_array, (1,3))
    #closed_image = closed_image.astype(clustered_array.dtype)

    sum_values2 = get_cluster_sum(cluster_centers, array, closed_image)
    return clustered_array, sum_values1, closed_image, sum_values2


def get_fluoro_intensity(image, num_peaks=3, min_distance=10, peak_radius=3, plot=False):
    distinctive_peaks = get_2d_peaks(image, min_distance, num_peaks)
    #TODO: add a check for the number of peaks
    # Create a mask to remove the area around the peaks
    mask = get_peak_mask(image, distinctive_peaks, peak_radius)
    # Interpolate the removed areas
    # x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    # image_interpolated = griddata((x[mask], y[mask]), image[mask], (x, y), method='nearest')
    interpolated_image = horizontal_axis_interpolation(image, mask)
    #interpolated_image = interpolated_image.astype('uint8')
    clustered_array, fluoro_intesity_sum1, closed_clusterd_array, fluoro_intesity_sum2 = kmean_cluster_2d_array(
        interpolated_image, n_clusters=2)

    if plot:
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image with Detected Peaks')
        plt.axis('off')
        for peak in distinctive_peaks:
            plt.plot(peak[1], peak[0], 'r+', markersize=15, markeredgewidth=2)
        # Plot the modified image
        plt.subplot(4, 1, 2)
        plt.imshow(interpolated_image, cmap='gray')
        plt.title('Image with removed peaks and interpolated')
        plt.axis('off')
        for peak in distinctive_peaks:
            plt.plot(peak[1], peak[0], 'r+', markersize=15, markeredgewidth=2)
        # Plot the clustered array
        plt.subplot(4, 1, 3)
        plt.imshow(clustered_array, cmap='viridis')
        plt.title(
            f'Clustered Array and fluorophore value is: {fluoro_intesity_sum1}')
        plt.subplot(4, 1, 4)
        plt.imshow(closed_clusterd_array, cmap='viridis')
        plt.title(
            f'Clustered Array after morphological operator and fluorophore value is: {fluoro_intesity_sum2}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return fluoro_intesity_sum1, fluoro_intesity_sum2


def get_peak_mask(image, peaks, peak_radius):
    mask = np.ones_like(image, dtype=bool)
    for peak in peaks:
        rr, cc = np.ogrid[:image.shape[0], :image.shape[1]]
        mask_area = (rr - peak[0])**2 + (cc - peak[1])**2 <= peak_radius**2
        mask[mask_area] = False
    return mask


def horizontal_axis_interpolation(image, mask):
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    # Interpolate only along the horizontal axis
    image_interpolated = np.zeros_like(image)
    for i in range(image.shape[0]):
        mask_row = mask[i, :]
        if np.any(mask_row):
            image_interpolated[i, :] = np.interp(
                x, x[mask_row], image[i, mask_row])
        else:
            image_interpolated[i, :] = image[i, :]
    return image_interpolated

def get_2d_peaks(image, min_distance=10, num_peaks=3):
    # Apply maximum filter
    neighborhood = filters.maximum_filter(image, size=min_distance)
    # Find local maxima
    local_max = (image == neighborhood)
    # Get the coordinates of the peaks
    peak_coords = np.argwhere(local_max)
    # Sort peaks by intensity
    peak_intensities = image[local_max]
    sorted_indices = np.argsort(peak_intensities)[::-1]
    sorted_peak_coords = peak_coords[sorted_indices]
    # Select the top num_peaks peaks
    distinctive_peaks = sorted_peak_coords[:num_peaks]
    return distinctive_peaks
