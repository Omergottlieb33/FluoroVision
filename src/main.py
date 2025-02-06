import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from tifffile import TiffFile
import matplotlib.pyplot as plt

from src.config.const import TIF_MAX_VAL, TIF_MIN_VAL, IMAGE_SIZE, SAVE_VIDEO, SAVE_HISOGRAM, SAVE_INTENSITY_MAP

from src.utils.plot_utils import save_histogram
from src.utils.math_utils import get_video_intensity_map
from src.utils.common_utils import yolo_results_to_dataframe, init_debug_logger

from src.tracking.tracker import BeadHorizontalTracker
from src.post_process.fluorophore_intensity import FluorophoreIntensityEstimator


class FluoroVision:
    def __init__(self, tif_path, weights, laser_map_path, tif_min_val=TIF_MIN_VAL, tif_max_val=TIF_MAX_VAL):
        self.tif_path = tif_path
        self.weights = weights
        self.laser_map_path = laser_map_path
        self.output_dir = os.path.join(
            os.path.dirname(self.tif_path), 'results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.tif_min_val = tif_min_val
        self.tif_max_val = tif_max_val
        self.init()

    def init(self):
        self.detection_model = YOLO(self.weights)
        self.estimator = FluorophoreIntensityEstimator(map_path=self.laser_map_path)
        self.video_path = os.path.join(self.output_dir, 'output_video.mp4')
        self.logger = init_debug_logger('debug', os.path.join(
            self.output_dir, 'debug.log'))
        self.save_tif_as_mp4(self.video_path)

    def estimate(self):
        self.df = self.detect_and_estimate_fluorophore_intensity()
        # initiate tracker
        self.tracker = BeadHorizontalTracker(self.df)
        # track beads
        self.tracker.track()
        self.tracker.assign_track_ids()

    def detect_and_estimate_fluorophore_intensity(self):
        df = pd.DataFrame()
        with TiffFile(self.tif_path) as tif:
            for i, page in tqdm(enumerate(tif.pages)):
                rgb_frame = self.tif_to_pil(page)
                detections = self.detection_model(rgb_frame, save_txt=False)
                box_list = detections[0].boxes.xywh.numpy()
                frame_df = yolo_results_to_dataframe(detections, i+1)
                tif_frame = page.asarray()
                fluoro_intesity_list, factor_list = [], []
                for j, box in enumerate(box_list):
                    fluoro_intesity, factor = self.estimator(i,
                        tif_frame, box, False, None)
                    fluoro_intesity_list.append(fluoro_intesity)
                    factor_list.append(factor)
                frame_df['fluoro_intensity'] = fluoro_intesity_list
                frame_df['factor'] = factor_list
                df = pd.concat([df, frame_df], ignore_index=True)
            return df

    def tif_to_pil(self, page):
        frame = np.clip((page.asarray().astype(np.float32) - self.tif_min_val) /
                        (self.tif_max_val - self.tif_min_val) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(frame)

    def save_results(self):
        self.df.to_csv(os.path.join(self.output_dir,
                       'tracked_results.csv'), index=False)
        if SAVE_VIDEO:
            result_video_path = os.path.join(self.output_dir, 'tracked_video.mp4')
            self.tracker.save_results_as_video(self.video_path, result_video_path)
        if SAVE_HISOGRAM:
            self.save_fluorophore_intesity_histogram()
        if SAVE_INTENSITY_MAP:
            self.save_bead_intensity_map()
        print('Results saved to:', self.output_dir)

    def save_tif_as_mp4(self, output_video_path, fps=10):
        with TiffFile(self.tif_path) as tif:
            frame_height, frame_width = tif.pages[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (frame_width, frame_height))

            for page in tqdm(tif.pages):
                frame = self.tif_to_pil(page)
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                video_writer.write(frame)

            video_writer.release()

    def save_fluorophore_intesity_histogram(self):
        # detection histogram
        self.df['refactored_fluoro_intensity'] = self.df['fluoro_intensity'] / \
            self.df['factor']
        fluoro_intensity = self.df['refactored_fluoro_intensity'].to_numpy()
        fluoro_intensity = fluoro_intensity.reshape(-1, 1)
        fluoro_intensity = fluoro_intensity[~np.isnan(fluoro_intensity)]
        save_histogram(fluoro_intensity, 25, 'Detection Fluorophore Intensity Histogram', 'Intensity',
                       'Frequency', os.path.join(self.output_dir, 'detection_fluorophore_intensity_histogram.png'))
        tracked_intensity = self.df.groupby(
            'track_id')['refactored_fluoro_intensity'].mean()
        save_histogram(tracked_intensity, 25, 'Tracked Fluorophore Intensity Histogram', 'Intensity',
                       'Frequency', os.path.join(self.output_dir, 'tracked_fluorophore_intensity_histogram.png'))

    def save_bead_intensity_map(self):
        path = os.path.join(self.output_dir, 'video_bead_intensity_map.png')
        z = len(self.df.groupby('track_id'))
        intensity_tensor = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], z))
        for i, (name, group) in enumerate(self.df.groupby('track_id')):
            for j, row in group.iterrows():
                if not np.isnan(row['refactored_fluoro_intensity']):
                    intensity_tensor[int(row['y_center']), int(
                        row['x_center']), i] = row['refactored_fluoro_intensity']
        intensity_map = get_video_intensity_map(intensity_tensor)
        plt.imshow(intensity_map, cmap='hot')
        plt.colorbar()
        plt.title('Bead Intensity Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(path)


if __name__ == '__main__':
    tiff_path = r'c:\Users\97254\Desktop\Resources\Technion\exploratory_resaerach\fluorovision\data\A1\2025_02_05_A1.tif'
    weights_path = r'C:\Users\97254\Desktop\git\FluoroVision\src\weights\yolo11n_bead_det_best_301224.pt'
    laser_map_path = r'c:\Users\97254\Desktop\Resources\Technion\exploratory_resaerach\fluorovision\data\mapV2.mat'
    fve = FluoroVision(tiff_path, weights_path, laser_map_path)
    fve.estimate()
    fve.save_results()
