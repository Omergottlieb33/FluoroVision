import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.common_utils import xywh_to_x1y1x2y2, get_location_factor
from src.config.const import IMAGE_SIZE


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

def save_histogram(array,bins, title, xlabel, ylabel, output_path=None):
    plt.hist(array, bins=bins, alpha=0.7, color='b')
    plt.plot([], [], label='Mean: {:.2f}'.format(np.mean(array)))
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path)
    plt.close()