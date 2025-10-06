from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomFlip,RandomRotation,RandomZoom

def standard_augs():
    """
    Standard augmentation stack for VAMML.
    """
    return Sequential([
        RandomFlip('horizontal_and_vertical'),
        RandomRotation(1,fill_mode="nearest"),
        RandomZoom(height_factor=0.1,width_factor=0.1,fill_mode="nearest"),
                      ])
