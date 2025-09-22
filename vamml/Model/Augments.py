from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomFlip,RandomRotation,RandomZoom
from tensorflow.random import get_global_generator
from tensorflow import cast,concat
import numpy as np

def other_augs():
    return Sequential([RandomFlip('horizontal_and_vertical'),
                       RandomRotation(1,fill_mode="nearest"),
                       RandomZoom(height_factor=0.1,width_factor=0.1,fill_mode="nearest"),
                      ])

def vamml_augments(input,output):
    """
    Custom augment function to apply a scaling factor to grayscale image output, with an inverse scaling factor to scalar input.
    Additionally pairs input and output images to equally apply other_augs to both before reconstructing input,output pairing.

    Args:
        input (([N,N] tensor, [M] tensor)) : A paired set of (1-channel image, data) with the first value in [M] corresponding to value to augment.
        output ([N,N] tensor)              : 1-channel image tensor.

    Returns:
        (([N,N] tensor, [M] tensor), ([N,N] tensor)) 
    """
    input_img = input[0]
    input_vals = input[1]
    # Augmenting dose/output to train network to recognize output scaling.
    gen = get_global_generator()
    scaler = gen.uniform([1,], minval = 0, maxval = 3)
    chance = gen.uniform([1,], minval = 0, maxval = 1)
    # Only applying augment 50% of the time to give clearer endpoint data for training
    scale = 1 + scaler*cast(chance > 0.5, np.float32)
    # Splitting and increasing dose by scale factor to represent increased dose relative to voxel intensity
    dose = input_vals[...,:1] * scale
    other = input_vals[...,1:]
    # Dividing output by scaling to represent decreased voxel stack intensity
    output_aug1 = output[0] / scale
    # Re-concatenating values
    aug_vals = concat([dose,other],-1)
    # Re-layering input with augmented output
    concat_img = concat([input_img, output_aug1], axis=-1)
    # Initializing standard augmentation stack
    aug = other_augs()
    # Limits augmentation to training batches, not validation
    img_aug = aug(concat_img, training=True)
    input_aug = img_aug[..., :1]
    output_aug2 = img_aug[..., 1:]
    return (input_aug,aug_vals), (output_aug2,)