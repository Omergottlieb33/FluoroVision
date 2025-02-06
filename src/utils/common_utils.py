import re
import numpy as np
import pandas as pd
import logging

def init_debug_logger(name, log_path=None, log_level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def xywh_to_x1y1x2y2(box, image_width, image_height):
    x_center, y_center, box_width, box_height = box
    abs_x_center = x_center * image_width
    abs_y_center = y_center * image_height
    abs_width = box_width * image_width
    abs_height = box_height * image_height
    # Calculate box corners
    x_min, y_min, x_max, y_max = xcycwh_to_x1y1x2y2(abs_x_center, abs_y_center, abs_width, abs_height)
    return x_min, y_min, x_max, y_max

def xcycwh_to_x1y1x2y2(xc,yc,width, height):
    x1 = xc - width/2
    y1 = yc - height/2
    x2 = xc + width/2
    y2 = yc + height/2
    return int(x1), int(y1), int(x2), int(y2)

def get_location_factor(x,y):
    p00 = 154.7
    p10 = 0.6552
    p01 = 0.5203
    p20 = -0.002902
    p11 = -0.002203
    p02 = -0.001476
    p30 = 3.383e-06
    p21 = 3.188e-06
    p12 = 1.472e-08
    p03 = -2.858e-07

    # Calculate the polynomial value
    val = (
        p00 +
        p10 * x + p01 * y +
        p20 * (x ** 2) + p11 * x * y + p02 * (y ** 2) +
        p30 * (x ** 3) + p21 * (x ** 2) * y + p12 * x * (y ** 2) + p03 * (y ** 3)
    )

    return val

def extract_frame_number(filename):
    """Extract the frame number from a filename like 'frame_0013.png'."""
    match = re.search(r'frame_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern.")

def yolo_results_to_dataframe(results, frame):
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
    return df
