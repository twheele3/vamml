import numpy as np
import io
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from skimage.measure import label
from scipy.ndimage import binary_erosion

def make_shape_mask(n,r,array_size,invert=True,**kwargs):
    '''
    n (int,list)     :   Number of key vertices around unit circle, or length-2 list-like with range
    r (float)[0,1]   :   Magnitude of perturbation
    invert (bool)    :   If true, produces False shape on True background
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
    arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))[:,:,0].astype(bool)
    io_buf.close()
    if invert:
        arr = np.invert(arr)
    return arr

def make_random_shape(random_state,shape_pars,hole_pars,hole_range,hole_count,hole_margin,**kwargs):
    """
    Makes random shapes based on distortion of vertices around a unit circle.
    """
    np.random.seed(random_state)
    arr = make_shape_mask(**shape_pars,**kwargs)
    margin = hole_margin
    shape_range = hole_range
    holes = hole_count
    for i in range(holes):
        # Generate random sized hole
        mask_size = np.random.randint(hole_range[0],hole_range[1])
        # Make negative boolean mask
        mask = make_shape_mask(**hole_pars,array_size=mask_size,invert=False)
    
        # Find bounding box of main shape to try to meaningfully overlap random hole location
        x_idx = np.where(arr.sum(axis=0))[0]
        xmin = x_idx[0] - margin
        xmax = x_idx[-1] + margin - mask_size
        xind = np.random.randint(xmin,xmax)
        y_idx = np.where(arr.sum(axis=1))[0]
        ymin = y_idx[0] - margin
        ymax = y_idx[-1] + margin - mask_size
        yind = np.random.randint(ymin,ymax)
    
        # Take boolean difference in target region 
        subset = arr[xind:xind+mask_size,yind:yind+mask_size]
        subset = subset * mask
        arr[xind:xind+mask_size,yind:yind+mask_size] = subset
    return arr

def clip_center(arr,px_radius = None, working_diameter = None, voxel_size = None,**kwargs):
    """Clips arr outside of circle of size px_radius from center of arr.
    Tries to calculate px_radius from other kwargs if not explicitly provided. 
    """
    if px_radius is None:
        px_radius = working_diameter / 2 / voxel_size
    mask = ((np.indices((arr.shape[0],arr.shape[1])) - arr.shape[0]/2)**2).sum(axis=0)**0.5 < px_radius
    return arr*mask

def circle_deform(random_state,**kwargs):
    arr = make_random_shape(random_state,**kwargs)
    arr = clip_center(arr,**kwargs)
    return arr

def rectangle_array(random_state,array_size,shape_pars,working_diameter,voxel_size,min_support,**kwargs):
    """Generates a set of rectangles falling within a working diameter to assemble a random shape
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

def rotate_point(v, origin, angle, **kwargs):
    angle = angle * np.pi / 180.0
    x = np.cos(angle) * (v[...,0]-origin) - np.sin(angle) * (v[...,1]-origin) + origin
    y = np.sin(angle) * (v[...,0]-origin) + np.cos(angle) * (v[...,1]-origin) + origin
    return np.array([x,y]).T

def continuity_test(arr):
    """Test whether there's only one labelable feature in an array"""
    return label(arr).max() == 1

def min_thickness_test(arr,min_thickness_kernel,**kwargs):
    """
    Test for minimum connecting thickness within continuous features.
    Detects if any holes open up in feature.

    Args:
        arr (2D numpy array) : Binary-coercable array
        min_thickness_kernel (int)     : Minimum feature radius (1/2 total thickness) to test for

    Returns:
        Boolean
    """
    dil_arr = binary_erosion(arr,np.ones((min_thickness_kernel,min_thickness_kernel)))
    return label(np.invert(dil_arr.astype(bool))).max() == label(np.invert(arr.astype(bool))).max()

def min_support_test(arr,min_support_kernel,**kwargs):
    """
    Test for minimum connecting thickness within continuous features.
    Detects if there's a change in number of discrete features after dilation based on minimum thickness.

    Args:
        arr (2D numpy array) : Binary-coercable array
        min_support_kernel (int)     : Minimum feature radius (1/2 total thickness) to test for

    Returns:
        Boolean
    """
    dil_arr = binary_erosion(arr,np.ones((min_support_kernel,min_support_kernel)))
    return label(dil_arr.astype(bool)).max() == 1