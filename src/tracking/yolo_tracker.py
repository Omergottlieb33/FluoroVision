import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def yolo_tracker_results_to_csv(results, csv_path):
    data = {
        'frame': [],
        'class': [],
        'conf': [],
        'x_center': [],
        'y_center': [],
        'width': [],
        'height': [],
        'track_id': []
    }
    for i, result in enumerate(results):
        frame = i + 1
        if result.boxes is not None:
            xywh_boxes = result.boxes.xywh.numpy()
            cls = result.boxes.cls.numpy()
            conf = result.boxes.conf.numpy()
            id = result.boxes.id.numpy() if result.boxes.is_track else np.full(len(cls), None)
            
            data['frame'].extend(np.full(len(cls), frame))
            data['class'].extend(cls)
            data['conf'].extend(conf)
            data['x_center'].extend(xywh_boxes[:, 0])
            data['y_center'].extend(xywh_boxes[:, 1])
            data['width'].extend(xywh_boxes[:, 2])
            data['height'].extend(xywh_boxes[:, 3])
            data['track_id'].extend(id)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    

def get_yolo_trakcing_results(weights_path, video_path ,results_path,save_video=False):
    """Track objects in a video using YOLO model."""
    # Load model
    model = YOLO(weights_path)
    # Track objects
    #TODO: test Byte tracker
    #TODO: check inputing pre conditions of movement of objects
    results = model.track(video_path, save=save_video)
    yolo_tracker_results_to_csv(results, results_path)
    return results

if __name__ == '__main__':
    weights_path = r'C:\Users\97254\Desktop\Resources\Technion\exploratory_resaerach\runs\train\exp\weights\best.pt'
    video_path = r'C:\Users\97254\Desktop\Resources\Technion\exploratory_resaerach\data\dataset\val_video.mp4'
    results_path = r'C:\Users\97254\Desktop\Resources\Technion\exploratory_resaerach\runs\detect\track\results.csv'
    results = get_yolo_trakcing_results(weights_path, video_path,results_path, save_video=False)
