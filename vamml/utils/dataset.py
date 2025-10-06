import os
import tensorflow as tf
import numpy as np
import numpy.typing as npt
import pandas as pd

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

def str_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode('utf-8')])
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_examples(df: pd.DataFrame, 
                 size: int = 256, 
                 create_example = None
                 ) -> dict[int,tf.train.Example]:
    # Nested iterator to assemble datasets across split
    """TODO: Document this"""

    if create_example is None:
        def create_example(image, shape, parameters):
            feature = {
                "image": float_feature_list(image),
                "label": float_feature_list(shape),
                "parameters" : float_feature_list(parameters),
            }
            return tf.train.Example(features=tf.train.Features(feature=feature))

    metadata_cols = [i for i in df.columns if i not in ['source','expt_index']]
    examples = {}
    for i in range(len(df)):
        features = df.iloc[i].to_dict()
        arr = load_npy_pair(features['source'],features['expt_index'])
        # TODO fix this redundant thign
        parameters = [features[i] for i in metadata_cols]
        # Normalizing arrays, pulling image:shape pair from first and last channels
        # TODO get a reshaping function working here
        image = arr[...,:1].astype(np.float32) / 255.
        shape = arr[...,-1:].astype(np.float32) / 255.
        examples[i] = create_example(image=image.flatten().tolist(), 
                                    shape=shape.flatten().tolist(), 
                                    parameters=parameters)
    return examples

def load_npy_pair(filename: str,
                  index: int,
                  axs: tuple[int,int] = (0,-1)) -> npt.NDArray:
    """Loads pickled numpy array (from numpy.save defaults, file.npy) to pull a subset, for 
    use in gathering indexed image data. The function presumes array.shape of 
    (N,M,M,X) where N is greater than index and X contains axs.  
    Returns array of (M,M,2) of the selected index and axs. Call order of axs assumes input is
    first value, output is second value, for sake of prepping data for training dataset.
    
    Args:
        filename (str) : Location of npy file to read
        index (int) : Index to read
        axs (tuple[int,int]) : (Input,Output) channels to read
    """
    arr = np.load(os.path.normpath(filename))[index]
    return np.concatenate([arr[...,axs[0]][...,None],arr[...,axs[1]][...,None]],axis=-1)