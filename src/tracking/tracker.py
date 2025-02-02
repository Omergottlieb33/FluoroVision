import os
import cv2
import json
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.common_utils import xcycwh_to_x1y1x2y2
from src.config.const import BEAD_MAX_AGE, FRAME_DIFF_THRESHOLD

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
        # box infornt of the previous box
        if prev_bbox[0] > bbox[0]:
            return 0
        # box is in the trajectory of the previous box
        if abs(prev_bbox[1] - bbox[1]) > self.pixel_height_threshold:
            return 0
        # box is in the tracking interval, ~8 frames
        if frame - self.frames[-1] > FRAME_DIFF_THRESHOLD:
            return 0
        # bead age condition
        if self.age > BEAD_MAX_AGE:
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
        for i, (frame, group) in tqdm(enumerate(self.df.groupby('frame'))):
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
    
    def save_results_as_figures(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            for i, (frame_id, group) in tqdm(enumerate(self.df.groupby('frame'))):
                if frame_id == frame_idx:
                    for _, row in group.iterrows():
                        x1, y1, x2, y2 = xcycwh_to_x1y1x2y2(row[3], row[4], row[5], row[6])
                        cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 1)
                        if row['track_id'] is not None:
                            cv2.putText(frame, f"{int(row['track_id'])}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            cv2.putText(frame, ' ', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f"{row['fluoro_intensity']:.2f}", (x1+10, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imwrite(f"{output_dir}/frame_{frame_idx}.png", frame)
            frame_idx += 1
        cap.release()
        cv2.destroyAllWindows()

    def save_results_as_video(self, video_path, output_video_path, fps=1):
        cap = cv2.VideoCapture(video_path)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_idx = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            for i, (frame_id, group) in tqdm(enumerate(self.df.groupby('frame'))):
                if frame_id == frame_idx:
                    for _, row in group.iterrows():
                        x1, y1, x2, y2 = xcycwh_to_x1y1x2y2(row[3], row[4], row[5], row[6])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        try:
                            cv2.putText(frame, f"{int(row['track_id'])}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        except:
                            print('invalid track_id')
                        try:
                            cv2.putText(frame, f"{int(row['fluoro_intensity']/row['factor'])}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        except:
                            print('Invalid fluoro_intensity')
            video_writer.write(frame)
            frame_idx += 1

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

    def assign_track_ids(self):
        track_ids = []
        for i, row in self.df.iterrows():
            frame = row['frame']
            bbox = row[3:7].to_numpy()
            assigned_id = None
            for tracker in self.trackers:
                if frame in tracker.frames:
                    idx = tracker.frames.index(frame)
                    tracked_bbox = tracker.boxes[idx]
                    if np.array_equal(bbox, tracked_bbox):
                        assigned_id = tracker.id
                        break
            track_ids.append(assigned_id)
        self.df['track_id'] = track_ids
