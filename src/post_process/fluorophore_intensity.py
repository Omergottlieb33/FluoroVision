import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
import scipy.ndimage.filters as filters

from src.config.const import BEAD_WIDTH_THRESHOLD, PEAK_INTENSITY_THRESHOLD, MIN_DISTANCE, NUM_PEAKS, PEAK_RADIUS, DISTANCE_FROM_EDGE
from src.utils.common_utils import xcycwh_to_x1y1x2y2
from src.utils.plot_utils import draw_bbox_on_frame
from src.utils.math_utils import get_dice_score

import logging

logger  = logging.getLogger('debug')

class FluorophoreIntensityEstimator:
    def __init__(self, map_path, bead_width_threshold=BEAD_WIDTH_THRESHOLD, peak_intensity_threshold=PEAK_INTENSITY_THRESHOLD,
                 min_distance=MIN_DISTANCE, num_peaks=NUM_PEAKS, peak_radius=PEAK_RADIUS):
        self.map = loadmat(map_path)['map']
        self.bead_width_threshold = bead_width_threshold
        self.peak_intensity_threshold = peak_intensity_threshold
        self.min_distance = min_distance
        self.num_peaks = num_peaks
        self.peak_radius = peak_radius
    
    def __call__(self, frame_id, frame, box, debug=False, save_path=None):
        xc, yc, w, h = box
        x1, y1, x2, y2 = xcycwh_to_x1y1x2y2(xc, yc, w, h)
        if (w < self.bead_width_threshold) or (x1 < DISTANCE_FROM_EDGE) or (x2 > frame.shape[1] - DISTANCE_FROM_EDGE):
            return np.nan, np.nan
        bead = frame[y1:y2, x1:x2]
        distinctive_peaks = self.get_2d_peaks(bead)
        # 2 peaks condition
        if len(distinctive_peaks) < self.num_peaks:
            logger.debug(f'{box} in frame {frame_id} has {len(distinctive_peaks)} peaks')
            return np.nan, np.nan
        # more than 4 peaks condition
        if len(distinctive_peaks) > 4:
            print('')
        xc, yc = self.get_bead_center(distinctive_peaks, x1, y1)
        factor = self.map[xc, yc]
        mask = self.get_peak_mask_circle(bead, distinctive_peaks)
        interpolated_image = self.horizontal_axis_interpolation(bead, mask)
        clustered_array, fluoro_intesity_sum1, closed_clusterd_array, fluoro_intesity_sum2 = self.kmean_cluster_2d_array(
            interpolated_image)
        optimal_mask = self.get_optimal_rectangle(closed_clusterd_array)
        optimal_rectangle_intensity = np.sum(interpolated_image * optimal_mask)
        if debug:
            frame_with_bbox = draw_bbox_on_frame(frame, (x1, y1, x2, y2))
            self.plot_steps(frame_with_bbox, bead, distinctive_peaks, interpolated_image, clustered_array, closed_clusterd_array,
                            optimal_mask, xc, yc, factor, fluoro_intesity_sum1, fluoro_intesity_sum2, optimal_rectangle_intensity, save_path)

        return optimal_rectangle_intensity, factor

    
    def get_2d_peaks(self, image):
        # Apply maximum filter
        neighborhood = filters.maximum_filter(image, size=self.min_distance)
        # Find local maxima
        local_max = (image == neighborhood)
        # Get the coordinates of the peaks
        peak_coords = np.argwhere(local_max)
        # Sort peaks by intensity
        peak_intensities = image[local_max]
        valid_peaks = peak_intensities > self.peak_intensity_threshold
        peak_coords = peak_coords[valid_peaks]
        peak_intensities = peak_intensities[valid_peaks]
        return peak_coords
    
    @staticmethod
    def get_bead_center(peaks, x1, y1):
        sorted_indices = np.argsort(peaks[:, 1])
        sorted_array = peaks[sorted_indices]
        xc, yc = int(x1 + sorted_array[1][1]), int(y1 + sorted_array[1][0])
        return xc, yc
    
    
    def get_peak_mask_circle(self, image, peaks):
        mask = np.ones_like(image, dtype=bool)
        for peak in peaks:
            rr, cc = np.ogrid[:image.shape[0], :image.shape[1]]
            mask_area = (rr - peak[0]) ** 2 + \
                (cc - peak[1]) ** 2 <= self.peak_radius ** 2
            mask[mask_area] = False
        return mask
    
    @staticmethod
    def horizontal_axis_interpolation(image, mask):
        x = np.arange(image.shape[1])
        y = np.arange(image.shape[0])
        image_interpolated = np.zeros_like(image)
        for i in range(image.shape[0]):
            mask_row = mask[i, :]
            if np.any(mask_row):
                image_interpolated[i, :] = np.interp(
                    x, x[mask_row], image[i, mask_row])
            else:
                image_interpolated[i, :] = image[i, :]
        return image_interpolated

    @staticmethod
    def kmean_cluster_2d_array(array):
        """Cluster 2D array into two classes using KMeans clustering and return the sum of the values in the cluster."""
        def get_cluster_sum(centers, image, cluster):
            label = np.argmax(centers)
            return np.sum(image[np.isin(cluster, label)])

        flattened_array = array.flatten().reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42).fit(flattened_array)
        clustered_array = kmeans.labels_.reshape(array.shape)
        cluster_centers = kmeans.cluster_centers_.flatten()
        if cluster_centers[0] > cluster_centers[1]:
            cluster_centers = cluster_centers[::-1]
            clustered_array = 1 - clustered_array
        sum_values1 = get_cluster_sum(cluster_centers, array, clustered_array)

        clustered_array = clustered_array.astype(array.dtype)
        kernel = np.ones((1, 3), np.uint8)
        closed_image = cv2.morphologyEx(
            clustered_array, cv2.MORPH_CLOSE, kernel)
        sum_values2 = get_cluster_sum(cluster_centers, array, closed_image)
        return clustered_array, sum_values1, closed_image, sum_values2
    
    @staticmethod
    def get_optimal_rectangle(cluster, height=[2, 3]):
        shape = cluster.shape
        non_zero_indices = np.where(cluster > 0)
        left_edge = np.min(non_zero_indices)
        right_edge = np.max(non_zero_indices)
        dice_score = 0
        optimal_mask = np.zeros(shape)
        for h in height:
            for i in range(shape[0]):
                if i + h > shape[0]:
                    break
                mask = np.zeros(shape)
                mask[i:i + h, left_edge:right_edge] = 1
                mask = mask.astype(cluster.dtype)
                dice_score_i = get_dice_score(cluster, mask)
                if dice_score_i > dice_score:
                    dice_score = dice_score_i
                    optimal_mask = mask
        return optimal_mask

   

    def plot_steps(self, frame, bead, peaks, interpolated_image, clustered_array, closed_clusterd_array, optimal_mask, xc, yc, factor, fluoro_intesity_sum1, fluoro_intesity_sum2, optimal_rectangle_intensity, save_path):
        fig, ax = plt.subplots(5, 2, figsize=(12, 12))
        ax[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('Frame with Bounding Box')
        ax[0, 0].axis('off')

        ax[2, 0].axis('off')
        ax[3, 0].axis('off')
        ax[0, 1].imshow(bead, cmap='gray')
        ax[0, 1].set_title('Original Image with Detected Peaks')
        ax[0, 1].axis('off')
        for peak in peaks:
            ax[0, 1].plot(peak[1], peak[0], 'r+',
                          markersize=15, markeredgewidth=2)

        ax[1, 1].imshow(interpolated_image, cmap='gray')
        ax[1, 1].set_title('Image with removed peaks and interpolated')
        ax[1, 1].axis('off')
        for peak in peaks:
            ax[1, 1].plot(peak[1], peak[0], 'r+',
                          markersize=15, markeredgewidth=2)
        ax[2, 1].imshow(clustered_array, cmap='viridis')
        ax[2, 1].set_title(
            f'Clustered Array and fluorophore value is: {(fluoro_intesity_sum1 / factor):.3f}')
        ax[3, 1].imshow(closed_clusterd_array, cmap='viridis')
        ax[3, 1].set_title(
            f'Clustered Array after morphological operator and fluorophore value is: {(fluoro_intesity_sum2 / factor):.3f}')
        ax[3, 1].axis('off')
        ax[4, 1].imshow(optimal_mask, cmap='viridis')
        ax[4, 1].set_title(
            f'Rectangle on the Image and its fluorophore value: {(optimal_rectangle_intensity / factor):.3f}')
        ax[4, 1].axis('off')
        ax[4, 0].axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()    
