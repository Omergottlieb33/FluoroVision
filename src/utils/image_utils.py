import numpy as np

def find_rectangle_coord(mask):
    """Find the closest coordinate to the top-left corner of an image in a binary mask."""
    # Find the indices of non-zero elements in the mask
    h, w = mask.shape
    non_zero_indices = np.where(mask > 0)
    if len(non_zero_indices[0]) == 0:
        return None, None
    # Get the coordinates of non-zero elements
    coords = np.column_stack(non_zero_indices)
    # Calculate the Euclidean distance to the top-left corner (0, 0)
    distances_top_left = np.sqrt(np.sum(coords**2, axis=1))
    # Find the index of the minimum distance
    min_distance_index = np.argmin(distances_top_left)
    max_distance_index = np.argmax(distances_top_left)
    # Get the coordinate with the minimum distance
    top_left_coord = tuple(coords[min_distance_index])
    bottom_right_coord = tuple(coords[max_distance_index])

    return top_left_coord[1], top_left_coord[0], bottom_right_coord[1], bottom_right_coord[0]