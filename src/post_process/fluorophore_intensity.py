import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.ndimage.filters as filters

from config.const import BEAD_WIDTH_THRESHOLD, PEAK_INTENSITY_THRESHOLD, MIN_DISTANCE, NUM_PEAKS, PEAK_RADIUS
from utils.common_utils import get_location_factor, xywh_to_x1y1x2y2
from utils.plot_utils import get_laser_intesity_facotr, draw_bbox_on_frame


def get_2d_array_peaks(image, min_distance=10, num_peaks=3):
    """Find the peaks in a 2D array and return the coordinates of the top num_peaks peaks."""

    neighborhood = filters.maximum_filter(image, size=min_distance)
    local_max = (image == neighborhood)
    peak_coords = np.argwhere(local_max)
    peak_intensities = image[local_max]
    sorted_indices = np.argsort(peak_intensities)[::-1]
    sorted_peak_coords = peak_coords[sorted_indices]

    return sorted_peak_coords[:num_peaks]


def kmean_cluster_2d_array(array, n_clusters=2):

    def morphological_operator(binary_image, size: tuple):
        kernel = np.ones(size, np.uint8)
        eroded_image = cv2.erode(binary_image, kernel, iterations=1)
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        return closed_image

    def get_cluster_sum(centers, image, cluster):
        # sorted_indices = np.argsort(centers)[::-1]
        # highest_value_clusters = sorted_indices[:2]
        label = np.argmax(centers)
        return np.sum(image[np.isin(cluster, label)])

    flattened_array = array.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flattened_array)
    clustered_array = kmeans.labels_.reshape(array.shape)
    cluster_centers = kmeans.cluster_centers_.flatten()
    sum_values1 = get_cluster_sum(cluster_centers, array, clustered_array)

    clustered_array = clustered_array.astype(array.dtype)
    closed_image = morphological_operator(clustered_array, (1, 3))
    # closed_image = closed_image.astype(clustered_array.dtype)

    sum_values2 = get_cluster_sum(cluster_centers, array, closed_image)
    return clustered_array, sum_values1, closed_image, sum_values2


def get_bead_fluoro_intensity(frame, box, num_peaks=3, min_distance=10, peak_radius=3, peak_th=1500, save=False, save_path=None):
    x1, y1, x2, y2 = xywh_to_x1y1x2y2(box, frame.shape[1], frame.shape[0])
    if x2 - x1 < BEAD_WIDTH_THRESHOLD:
        return np.nan, np.nan
    bead = frame[y1:y2, x1:x2]
    distinctive_peaks = get_2d_peaks(bead, min_distance, num_peaks, peak_th)
    if len(distinctive_peaks) < num_peaks:
        return np.nan, np.nan
    xc, yc = get_bead_center(distinctive_peaks, x1, y1)
    factor = get_location_factor(xc, yc)
    mask = get_peak_mask(bead, distinctive_peaks, peak_radius)
    interpolated_image = horizontal_axis_interpolation(bead, mask)
    clustered_array, fluoro_intesity_sum1, closed_clusterd_array, fluoro_intesity_sum2 = kmean_cluster_2d_array(
        interpolated_image, n_clusters=2)
    if save:
        frame_with_bbox = draw_bbox_on_frame(frame, (x1, y1, x2, y2))
        intensity_map = get_laser_intesity_facotr()
        fig, ax = plt.subplots(4, 2, figsize=(12, 12))
        ax[0, 0].imshow(cv2.cvtColor(frame_with_bbox, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('Frame with Bounding Box')
        ax[0, 0].axis('off')
        # ax[2, 0].imshow(intensity_map, cmap='viridis')
        im10 = ax[1, 0].imshow(intensity_map, cmap='viridis')
        ax[1, 0].plot(xc, yc, 'r+', markersize=15,
                      markeredgewidth=2, label=f'Bead Center')
        ax[1, 0].set_title(f'Laser Intensity Factor Map')
        ax[1, 0].axis('off')
        fig.colorbar(im10, ax=ax[1, 0])

        ax[2, 0].axis('off')
        ax[3, 0].axis('off')
        ax[0, 1].imshow(bead, cmap='gray')
        ax[0, 1].set_title('Original Image with Detected Peaks')
        ax[0, 1].axis('off')
        for peak in distinctive_peaks:
            ax[0, 1].plot(peak[1], peak[0], 'r+',
                          markersize=15, markeredgewidth=2)

        ax[1, 1].imshow(interpolated_image, cmap='gray')
        ax[1, 1].set_title('Image with removed peaks and interpolated')
        ax[1, 1].axis('off')
        for peak in distinctive_peaks:
            ax[1, 1].plot(peak[1], peak[0], 'r+',
                          markersize=15, markeredgewidth=2)
        # Plot the clustered array
        ax[2, 1].imshow(clustered_array, cmap='viridis')
        ax[2, 1].set_title(
            f'Clustered Array and fluorophore value is: {(fluoro_intesity_sum1/factor):.3f}')
        ax[3, 1].imshow(closed_clusterd_array, cmap='viridis')
        ax[3, 1].set_title(
            f'Clustered Array after morphological operator and fluorophore value is: {(fluoro_intesity_sum2/factor):.3f}')
        ax[3, 1].axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        # plt.show()
        plt.close()
    return fluoro_intesity_sum1/factor, fluoro_intesity_sum2/factor


def get_bead_center(peaks, x1, y1):
    sorted_indices = np.argsort(peaks[:, 1])
    sorted_array = peaks[sorted_indices]
    xc, yc = int(x1+sorted_array[1][1]), int(y1+sorted_array[1][0])
    return xc, yc


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


def get_2d_peaks(image, min_distance=10, num_peaks=3, peak_th=1500):
    # Apply maximum filter
    neighborhood = filters.maximum_filter(image, size=min_distance)
    # Find local maxima
    local_max = (image == neighborhood)
    # Get the coordinates of the peaks
    peak_coords = np.argwhere(local_max)
    # Sort peaks by intensity
    peak_intensities = image[local_max]
    valid_peaks = peak_intensities > peak_th
    peak_coords = peak_coords[valid_peaks]
    peak_intensities = peak_intensities[valid_peaks]

    # sorted_indices = np.argsort(peak_intensities)[::-1]
    # sorted_peak_coords = peak_coords[sorted_indices]
    # # Select the top num_peaks peaks+
    # distinctive_peaks = sorted_peak_coords[:num_peaks]

    return peak_coords
