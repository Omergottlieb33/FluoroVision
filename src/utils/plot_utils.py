import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.common_utils import xywh_to_x1y1x2y2
from post_process.fluorophore_intensity import get_location_factor
from config.const import IMAGE_SIZE


def show_frame_with_annotations(frame, box_list, title):
    height, width = frame.shape[:2]
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if box_list is not None:
        for box in box_list:
            x_min, y_min, x_max, y_max = xywh_to_x1y1x2y2(box, width, height)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    plt.title(title)
    plt.show()

def plot_laser_intesity_facotr():
    intensity_map = np.zeros(IMAGE_SIZE)
    for x in range(0,IMAGE_SIZE[0]-1):
        for y in range(0,IMAGE_SIZE[1]-1):
            intensity_map[x,y] = get_location_factor(x,y)
    plt.imshow(intensity_map, cmap='viridis')
    plt.colorbar()
    plt.title('Laser Intensity Factor')
    plt.show()