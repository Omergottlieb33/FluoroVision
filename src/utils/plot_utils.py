import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.common_utils import xywh_to_x1y1x2y2, get_location_factor
from src.config.const import IMAGE_SIZE


def show_frame_with_annotations(frame, box_list, title):
    height, width = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if box_list is not None:
        for box in box_list:
            x_min, y_min, x_max, y_max = xywh_to_x1y1x2y2(box, width, height)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    return frame

def draw_bbox_on_frame(frame, box, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = box
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame
    

def get_laser_intesity_facotr():
    intensity_map = np.zeros(IMAGE_SIZE)
    for x in range(0,IMAGE_SIZE[0]-1):
        for y in range(0,IMAGE_SIZE[1]-1):
            intensity_map[x,y] = get_location_factor(x,y)
    return intensity_map

def draw_yolo_results_on_frame(results, output_dir):
    for i, result in enumerate(results):
        img_path = os.path.join(output_dir, f'frame_{i:04d}.png')
        result.save(img_path)