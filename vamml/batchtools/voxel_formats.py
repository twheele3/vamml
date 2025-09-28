import os

import numpy as np
import numpy.typing as npt
from .utils import load_json,save_json

def export_tomolite(arr: npt.NDArray[np.uint8],
             export_dir: str, 
             export_name: str = 'voxels.dat',
             meta_template: str|None = None,
             metadata: dict|None = None,
             **kwargs) -> None:
    """Exports voxels and metadata to base_dir/voxels according to Tomolite printer format. 
    Requires template file of voxels.meta.txt to modify. Otherwise outputs key pars to info.txt.

    Args:
        arr (np.array[z,xy,xy]) : A 3D uint8 array of voxels to export.
        export_dir (str) : Export directory for voxels.dat, and voxels.meta.txt if meta_template specified
        meta_template (str) : Location of template voxels.meta.txt. Default: None
        metadata (dict) : key:value pairs to apply to voxels.meta.txt
    """

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    arr.tofile(os.path.join(export_dir,export_name))
    if meta_template is not None:
        try:
            d = load_json(meta_template)
        except:
            raise ValueError(f'Unable to load file as json: {meta_template}')
        if metadata is not None:
            # Removing any extraneous parameters in case they break anything on-server.
            for k in [i for i in metadata.keys()]:
                if k not in d.keys():
                    del metadata[k]
            d.update(metadata)
            # Reiterating voxel_size down further to be safe
            if 'voxel_size' in metadata.keys():
                d['metadata']['algorithm']['voxel_size'] = metadata['voxel_size']
                # Calculating position based on assumed method of initial calculation
                offset = -1 * metadata['voxel_size']*arr.shape[2] / 2
                d['position'] = [offset,offset,0.0]
        # Adding r_calculated based on assumption of necessity for calculation, and how it does it
        d['metadata']['r_calculated'] = np.where(arr.sum((1,2))>0)[0]
        d['shape'] = arr.shape
        save_json(os.path.join(export_dir,'.'.join([os.path.splitext(export_name)[0],'meta','txt'])),d)
    else:
        d = {'shape':arr.shape}
        if type(metadata) is dict:
            d.update(metadata)
        save_json(os.path.join(export_dir,'.'.join([' '.join([os.path.splitext(export_name)[0],'info']),'txt'])),d)

def import_tomolite(src_dat: str,
                    src_meta: str|None,
                    shape: tuple[int,int,int]|None,
                    **kwargs) -> npt.NDArray:
    """Load voxels.dat and voxels.meta.txt into array. Can also try from pars.json, or try to infer size.
    
    Args:
        src_dat (str) : voxels.dat file to load.
        src_meta (str|None) : Either voxels.meta.txt or pars.json file associated with array.
                              If None, uses shape or infers. Default None.
        shape (tuple[int,int,int]|None) : Shape of array, if known. If None, and no src_meta, infers. Default None.
        
    Returns:
        np.array([Z,X,Y],dtype=np.uint8)"""
    arr = np.fromfile(src_dat, dtype=np.uint8)
    # Pulling shape from json file if not given
    if (type(src_meta) is str) and (shape is None):
        try:
            d = load_json(src_meta)
        except:
            raise ValueError(f'Unable to load file as json: {src_meta}')
        if 'shape' in d.keys():
            if (type(d['shape']) is list) and (len(d['shape']) == 3):
                shape = d['shape']
        elif ('array_size' in d.keys()):
            if type(d['shape']) is int:
                a = d['array']
                shape = [len(arr)//a**2,a,a]
    # Inferring array size based on assumption that array is about twice as high as wide or deep
    if shape is None:
        a = np.arange(32,513)
        candidates = np.where(((len(arr) / a**2)% 1)==0)[0] + 32
        top3 = np.abs((candidates / (len(arr) // candidates**2))-0.5).argsort()
        shape = [len(arr)//candidates[top3[0]],candidates[top3[0]],candidates[top3[0]]]
        second = [len(arr)//candidates[top3[1]],candidates[top3[1]],candidates[top3[1]]]
        third = [len(arr)//candidates[top3[2]],candidates[top3[2]],candidates[top3[2]]]
        print(f'Most likely shape inferred is {shape}')
        print(f'Other likely candidates are {second} or {third}')
    return arr.reshape(shape)
    


