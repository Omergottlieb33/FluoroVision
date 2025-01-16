import numpy as np

def get_dice_score(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2)
    dice = 2 * intersection / union
    return dice