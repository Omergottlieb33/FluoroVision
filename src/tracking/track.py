import sys
sys.path.append(r'C:\Users\97254\Desktop\git\sort')

import os
import numpy as np
from sort import Sort
from tqdm import tqdm
from ultralytics import YOLO


from src.utils.common_utils import yolo_results_to_dataframe
from src.utils.plot_utils import draw_yolo_results_on_frame

from src.tracking.tracker import BeadHorizontalTracker

def yolo_resuls_to_sort_format(detections):
    numpy_detections = {}
    for i, frame_detections in tqdm(enumerate(detections)):
        boxes = frame_detections.boxes.xyxy.numpy()
        confidence = frame_detections.boxes.conf.numpy()
        numpy_detections[str(i)] = np.hstack((boxes, confidence.reshape(-1,1)))
    return numpy_detections

def track_beads(video_path, weights, min_hits=1, output_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(weights)
    #tracker = Sort(min_hits=min_hits)
    detections = model(video_path, save_txt=False)
    #draw_yolo_results_on_frame(detections, output_dir)
    df = yolo_results_to_dataframe(detections)
    df.to_csv(os.path.join(output_dir, 'det_results.csv'), index=False)
    tracker = BeadHorizontalTracker(df)
    tracker.track()
    tracker.save_results(os.path.join(output_dir, 'tracked_results.json'))
    tracker.print_results(video_path, output_dir)
    #numpy_detections = yolo_resuls_to_sort_format(detections)
    




if __name__ == '__main__':
    weights = r'C:\Users\97254\Desktop\git\FluoroVision\src\weights\yolo11n_bead_det_best_301224.pt'
    video_path = r'C:\Users\97254\Desktop\Resources\Technion\exploratory_resaerach\data\dataset\val_video.mp4'
    output_dir = r'C:\Users\97254\Desktop\Resources\Technion\exploratory_resaerach\runs\detect\track\horizontal_tracking_v1'
    track_beads(video_path, weights, output_dir=output_dir)





