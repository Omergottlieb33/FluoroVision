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
