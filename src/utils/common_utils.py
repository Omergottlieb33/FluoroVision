def xywh_to_x1y1x2y2(box, image_width, image_height):
    x_center, y_center, box_width, box_height = box
    abs_x_center = x_center * image_width
    abs_y_center = y_center * image_height
    abs_width = box_width * image_width
    abs_height = box_height * image_height
    # Calculate box corners
    x_min = abs_x_center - abs_width / 2
    y_min = abs_y_center - abs_height / 2
    x_max = abs_x_center + abs_width / 2
    y_max = abs_y_center + abs_height / 2
    return int(x_min), int(y_min), int(x_max), int(y_max)
