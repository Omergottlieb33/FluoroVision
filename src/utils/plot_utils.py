import cv2
import matplotlib.pyplot as plt
from src.utils.common_utils import xywh_to_x1y1x2y2


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