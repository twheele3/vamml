import io

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from skimage.measure import label
from scipy.ndimage import binary_erosion

def make_shape_mask(n: int|tuple[int,int],
                    r: float,
                    array_size: int,
                    invert: bool=True,
                    **kwargs 
                    ) -> npt.NDArray[np.bool]:
    '''Generates boolean shape mask by deforming vertices describing a circle with random
    perturbations.

    Args:
        n (int,list)     :   Number of key vertices around unit circle, or length-2 list-like with range
        r (float)[0,1]   :   Magnitude of perturbation
        array_size (int) :   Size of 2DArray axes
        invert (bool)    :   If true, produces False shape on True background

    Returns:
        np.array(tuple[array_size,array_size], np.dtype[np.bool])
    '''
    shape = (array_size,array_size)
    if type(n) != int:
        n_use = np.random.randint(n[0],n[1])
    else:
        n_use = n
    if (type(r) != int) & (type(r) != float):
        r_use = np.random.random(1)[0]*(r[1]-r[0]) + r[0]
    else:
        r_use = r
    
    N = n_use*3+1 # number of points in the Path
    angles = np.linspace(0,2*np.pi,N)
    codes = np.full(N,Path.CURVE4)
    codes[0] = Path.MOVETO
    
    verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r_use*np.random.random(N)+1-r_use)[:,None]
    verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    path = Path(verts, codes)
    
    fig = plt.figure(figsize=shape,dpi=1)
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='k', lw=2)
    ax.add_patch(patch)
    
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off') # removes the axis to leave only the shape
    
    io_buf = io.BytesIO()
    plt.close(fig)
    fig.savefig(io_buf, format='raw', dpi=1)
    io_buf.seek(0)
    arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8).reshape(
        (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:,:,0].astype(bool)
    io_buf.close()
    if invert:
        arr = np.invert(arr)
    return arr

def make_random_shape(random_state: int,
                      shape_pars: dict,
                      hole_pars: dict,
                      hole_range: tuple[int,int],
                      hole_count: int,
                      hole_margin: int,
                      **kwargs
                      ) -> npt.NDArray[np.bool]:
    """
    Makes random shapes based on distortion of vertices around a unit circle.

    Args:
        random_state (int)  : Seed for random generator
        shape_pars (dict)   : Dictonary of arguments for make_shape_mask
        hole_pars (dict)   : Dictonary of arguments for make_shape_mask for negative elements
        hole_range (tuple-like[int,int]) : Range of sizes hole could be.
        hole_count (int)   : Max amount of holes in object
        hole_margin (int)  : Distance from edge of base circle to enforce hole positions

    Returns:
        np.array(tuple[array_size,array_size], np.dtype[np.bool])
    """
    np.random.seed(random_state)
    arr = make_shape_mask(**shape_pars,**kwargs)
    for i in range(hole_count):
        # Generate random sized hole
        mask_size = np.random.randint(hole_range[0],hole_range[1])
        # Make negative boolean mask
        mask = make_shape_mask(**hole_pars,array_size=mask_size,invert=False)
    
        # Find bounding box of main shape to try to meaningfully overlap random hole location
        x_idx = np.where(arr.sum(axis=0))[0]
        xmin = x_idx[0] - hole_margin
        xmax = x_idx[-1] + hole_margin - mask_size
        xind = np.random.randint(xmin,xmax)
        y_idx = np.where(arr.sum(axis=1))[0]
        ymin = y_idx[0] - hole_margin
        ymax = y_idx[-1] + hole_margin - mask_size
        yind = np.random.randint(ymin,ymax)
    
        # Take boolean difference in target region 
        subset = arr[xind:xind+mask_size,yind:yind+mask_size]
        subset = subset * mask
        arr[xind:xind+mask_size,yind:yind+mask_size] = subset
    return arr

def clip_center(arr: npt.NDArray,
                px_radius: int|None = None, 
                working_diameter: float|None = None,
                voxel_size: float|None = None,
                **kwargs
                ) -> npt.NDArray:
    """Clips arr outside of circle of size px_radius from center of arr.
    Tries to calculate px_radius from other kwargs if not explicitly provided.

    Args:
        arr (np.array[N,N,...]) : Numpy array where first two dimensions are assumed equal
        px_radius (int,None) : Radius in pixels to mask outside of. Calculated from other 
                               values if None. Default None.
        working_diameter (float,None) : Diameter in units/px, used if px_radius=None. Default None.
        voxel_size (float,None) : Voxel calibration in units/px, used if px_radius=None. Default None.

    Returns:
        np.array[N,N,...]
    """
    if px_radius is None:
        px_radius = working_diameter / 2 / voxel_size
    mask = ((np.indices((arr.shape[0],arr.shape[1])) - arr.shape[0]/2)**2).sum(axis=0)**0.5 < px_radius
    while mask.ndim < arr.ndim:
        mask = np.expand_dims(mask,axis=-1)
    # TODO: make sure this typing didn't break anything
    return arr*mask.astype(arr.dtype)

def circle_deform(random_state: int,
                  **kwargs
                  ) -> npt.NDArray:
    """ Method for making random shapes based on deforming circles.

    Args:
        random_state (int) : Seed for generator
        **kwargs (Experiment.pars) : From Experiment.pars structure
    
    Returns:
        np.array((N,N),bool)
    """
    arr = make_random_shape(random_state,**kwargs)
    arr = clip_center(arr,**kwargs)
    return arr

def rectangle_array(random_state: int,
                    array_size: int,
                    shape_pars: dict,
                    working_diameter: float,
                    voxel_size: float,
                    min_support: float,
                    **kwargs
                    ) -> npt.NDArray:
    """Generates a set of rectangles falling within a working diameter to assemble a random shape.

    Args:
        random_state (int)  : Seed for random generator
        array_size (int) :   Size of 2DArray axes
        shape_pars (dict)   : Dictonary of arguments for assembly
        working_diameter (float) : Diameter in units
        voxel_size (float) : Voxel calibration in units/px
        min_support (float)  : Minimum thickness of elements in units 

    Returns:
        np.array(tuple[array_size,array_size], np.dtype[np.bool])
    """
    rng = np.random.default_rng(random_state)
    px_radius = working_diameter / 2 / voxel_size
    min_width = min_support / voxel_size
    centroid = array_size//2
    squares = []
    # Generating random blocks.
    for i in range(shape_pars['n']):
        idx = np.array([[-1000,-1000],[-1000,-1000]])
        while np.any(((centroid - idx)**2).sum(1) > px_radius**2) | np.any(np.abs(np.diff(idx,axis=0)) < min_width):
            start = rng.integers(centroid, centroid+px_radius, 2)[None,...]
            end = start - rng.integers(0, px_radius*2 * shape_pars['r'], 2)[None,...]
            idx = np.concatenate([start,end])
        idx.sort(axis=0)
        rotation = rng.integers(0,shape_pars['cardinality'])
        
        idx = rotate_point(idx,centroid,360*rotation/shape_pars['cardinality']).astype(int)
        idx.sort(axis=0)
        squares.append(idx)
    
    arr = np.zeros((array_size,array_size),dtype = bool)
    for idx in squares:
        arr[idx[0,0]:idx[1,0]+1,idx[0,1]:idx[1,1]+1] = True
    return arr

def rotate_point(v: npt.ArrayLike, 
                 origin: tuple, 
                 angle: float
                 ) -> npt.NDArray:
    """Rotates series of x,y points around origin.
    
    Args:
        v (NDArray[...,2]) : NDArray with last dimension representing x,y coordinates.
        origin (tuple[2]) : Tuple describing x,y origin coordinates.
        angle (float)     : Angle to rotate in degrees
    Returns:
        NDArray
    """
    angle = angle * np.pi / 180.0
    x = np.cos(angle) * (v[...,0]-origin) - np.sin(angle) * (v[...,1]-origin) + origin
    y = np.sin(angle) * (v[...,0]-origin) + np.cos(angle) * (v[...,1]-origin) + origin
    return np.array([x,y]).T

def continuity_test(arr: npt.NDArray) -> bool:
    """Test whether there's only one labelable feature in an array"""
    return label(arr).max() == 1

def min_thickness_test(arr: npt.NDArray,
                       min_thickness_kernel: int,
                       **kwargs
                       ) -> bool:
    """
    Test for minimum connecting thickness within continuous features.
    Detects if no gaps open up in eroded features. Returns True if no new gaps. 

    Args:
        arr (2D numpy array) : Binary-coercable array
        min_thickness_kernel (int)     : Minimum feature radius (1/2 total thickness) to test for

    Returns:
        Boolean
    """
    dil_arr = binary_erosion(arr,np.ones((min_thickness_kernel,min_thickness_kernel)))
    return label(np.invert(dil_arr.astype(bool))).max() == label(np.invert(arr.astype(bool))).max()

def min_support_test(arr: npt.NDArray,
                     min_support_kernel: int,
                     **kwargs
                     ) -> bool:
    """
    Test for minimum connecting thickness within continuous features.
    Detects if there's a change in number of discrete features after 
    dilation based on minimum thickness. Returns True if no change.

    Args:
        arr (2D numpy array) : Binary-coercable array
        min_support_kernel (int)     : Minimum feature radius (1/2 total thickness) to test for

    Returns:
        Boolean
    """
    dil_arr = binary_erosion(arr,np.ones((min_support_kernel,min_support_kernel)))
    return label(dil_arr.astype(bool)).max() == 1