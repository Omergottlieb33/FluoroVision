import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from tifffile import TiffFile

from src.post_process.fluorophore_intensity import FluorophoreIntensityEstimator
from src.tracking.tracker import BeadHorizontalTracker
from src.utils.common_utils import yolo_results_to_dataframe
from src.config.const import TIF_MAX_VAL, TIF_MIN_VAL


class FluoroVisionExstimatior:
    def __init__(self, tif_path, weights, tif_min_val=TIF_MIN_VAL, tif_max_val=TIF_MAX_VAL):
      self.tif_path = tif_path
      self.weights = weights
      self.output_dir = os.path.join(os.path.dirname(self.tif_path), 'results')
      os.makedirs(self.output_dir, exist_ok=True)
      self.tif_min_val = tif_min_val
      self.tif_max_val = tif_max_val
      self.init()
    
    def init(self):
      self.detection_model = YOLO(self.weights)
      self.estimator = FluorophoreIntensityEstimator()
    
    def run(self):
       self.df = self.detect_and_estimate_fluorophore_intensity()
       #TODO: add multiple boxes logic on same image height (kalman, frame rate, fluid velocity, etc.)
       tracker = BeadHorizontalTracker(self.df)
       tracker.track()
       tracker.save_results(os.path.join(self.output_dir, 'tracked_results.json'))
    
    def detect_and_estimate_fluorophore_intensity(self):
       df = pd.DataFrame()
       with TiffFile(self.tif_path) as tif:
        for i, page in tqdm(enumerate(tif.pages)):
            rgb_frame = self.tif_to_pil(page)
            detections = self.detection_model(rgb_frame, save_txt=False)
            box_list = detections[0].boxes.xywh.numpy()
            frame_df = yolo_results_to_dataframe(detections, i+1)
            tif_frame = page.asarray()
            fluoro_intesity_list = []
            for j, box in enumerate(box_list):
                fluoro_intesity = self.estimator(tif_frame, box, False, None)
                fluoro_intesity_list.append(fluoro_intesity)
            frame_df['fluoro_intensity'] = fluoro_intesity_list
            df = pd.concat([df, frame_df], ignore_index=True)
        return df
        
    def tif_to_pil(self, page):
       frame = np.clip((page.asarray().astype(np.float32) - self.tif_min_val) / (self.tif_max_val - self.tif_min_val) * 255, 0, 255).astype(np.uint8)
       return  Image.fromarray(frame)
       
    

if __name__ == '__main__':
    tiff_path = r'C:\Users\97254\Desktop\Resources\Technion\exploratory_resaerach\fluorovision\data\AB3C\AB3C.tif'
    weights_path = r'C:\Users\97254\Desktop\git\FluoroVision\src\weights\yolo11n_bead_det_best_301224.pt'
    fve = FluoroVisionExstimatior(tiff_path, weights_path)
    fve.run()