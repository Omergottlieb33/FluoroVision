import numpy as np

def get_dice_score(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    dice = 2 * intersection / union
    return dice

def get_video_intensity_map(intensity_map):
    # Initialize the result array with zeros
    result = np.zeros((intensity_map.shape[0], intensity_map.shape[1]))
    # Iterate over each position in the 2D plane
    for i in range(intensity_map.shape[0]):
        for j in range(intensity_map.shape[1]):
            # Extract the values along the z-axis at position (i, j)
            values = intensity_map[i, j, :]
            # Count the number of non-zero values
            non_zero_values = values[values != 0]
            if len(non_zero_values) > 1:
                # If more than one non-zero value, average them
                result[i, j] = np.mean(non_zero_values)
            else:
                # Otherwise, sum the values
                result[i, j] = np.sum(values)
    
    return result