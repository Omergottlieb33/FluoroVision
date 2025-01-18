import os
import cv2
import json
import codecs
import numpy as np
import pandas as pd

from src.utils.common_utils import xcycwh_to_x1y1x2y2

class BoxTracker:
    def __init__(self, id, bbox, frame, pixel_height_threshold=2):
        self.id = id
        self.boxes = [bbox]  # (xc, yc, w, h)
        self.frames = [frame]
        self.age = 0
        self.time_since_update = 0
        self.pixel_height_threshold = pixel_height_threshold
        
    
    def update(self, bbox, frame):
        prev_bbox = self.boxes[-1]
        if prev_bbox[0] > bbox[0]:
            return 0
        if abs(prev_bbox[1] - bbox[1]) > self.pixel_height_threshold:
            return 0
        self.boxes.append(bbox)
        self.frames.append(frame)
        self.age += 1
        self.time_since_update = 0
        return 1


class BeadHorizontalTracker:
    """
    A class to track beads in a horizontal movement under the assumption that the beads are moving in a straight line
    """
    def __init__(self, df, detection_threshold=0.2):
        self.df = df
        self.detection_threshold = detection_threshold
        self.trackers = []
    
    def track(self):
        id = 0
        for i, (frame, group) in enumerate(self.df.groupby('frame')):
            if i == 0:
                for _, row in group.iterrows():
                    tracker = BoxTracker(id, row[3:7].to_numpy(), frame)
                    id += 1
                    self.trackers.append(tracker)
            else:
                matched_idx = []
                for j, row in group.iterrows():
                    for tracker in self.trackers:
                        updated = tracker.update(row[3:7].to_numpy(), frame)
                        if updated:
                            matched_idx.append(j)
                            break
                unmatched_idx = list(set(group.index) - set(matched_idx))
                for idx in unmatched_idx:
                    tracker = BoxTracker(id, self.df.iloc[idx, 3:7].to_numpy(), frame)
                    id += 1
                    self.trackers.append(tracker)
    
    def save_results(self, output_file):
        results = []
        for tracker in self.trackers:
            results.append({
                'id': tracker.id,
                'frames': tracker.frames,
                'boxes': np.array(tracker.boxes).astype(float).tolist()
            })
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4) ### this saves the array in .json format
    
    def print_results(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            for tracker in self.trackers:
                if frame_idx in tracker.frames:
                    idx = tracker.frames.index(frame_idx)
                    bbox = tracker.boxes[idx]
                    x1, y1, x2, y2 = xcycwh_to_x1y1x2y2(bbox[0], bbox[1], bbox[2], bbox[3])
                    cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{tracker.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(f"{output_dir}/frame_{frame_idx}.png", frame)
            frame_idx += 1
        cap.release()
        cv2.destroyAllWindows()
        