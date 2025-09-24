import numpy as np
import io
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from skimage.measure import label
from scipy.ndimage import binary_erosion

def make_shape_mask(n,r,shape,cardinality=None,invert=True):
    '''
    n (int)          :   Number of key vertices around unit circle
    r (float)[0,1]   :   Magnitude of perturbation
    shape (int)      :   x,y dimensions of image to make, in pixels
    invert (bool)    :   If true, produces False shape on True background
    '''
    shape = (shape,shape)
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

def make_random_shape(pars,random_state):
    """
    Makes random shapes based on distortion of vertices around a unit circle.
    """
    np.random.seed(random_state)
    arr = make_shape_mask(**pars['shape_pars'])
    margin = pars['hole_margin']
    shape_range = pars['hole_range']
    holes = pars['hole_count']
    for i in range(holes):
        # Generate random sized hole
        mask_size = np.random.randint(pars['hole_range'][0],pars['hole_range'][1])
        # Make negative boolean mask
        mask = make_shape_mask(**pars['hole_pars'],shape=mask_size,invert=False)
    
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

def clip_center(arr,pars):
    px_radius = pars['working_diameter'] / 2 / pars['voxel_size']
    frame_size = pars['shape_pars']['shape']
    mask = ((np.indices((frame_size,frame_size)) - frame_size/2)**2).sum(axis=0)**0.5 < px_radius
    return arr*mask

def continuity_test(arr):
    return label(arr).max() == 1

def circle_deform(pars,random_state):
    arr = make_random_shape(pars,random_state)
    arr = clip_center(arr,pars)
    return arr

def rectangle_array(pars,random_state):
    rng = np.random.default_rng(random_state)
    px_radius = pars['working_diameter'] / 2 / pars['voxel_size']
    min_width = pars['min_support'] / pars['voxel_size']
    centroid = pars['shape_pars']['shape']//2
    squares = []
    # Generating random blocks.
    for i in range(pars['shape_pars']['n']):
        idx = np.array([[-1000,-1000],[-1000,-1000]])
        while np.any(((centroid - idx)**2).sum(1) > px_radius**2) | np.any(np.abs(np.diff(idx,axis=0)) < min_width):
            start = rng.integers(centroid, centroid+px_radius, 2)[None,...]
            end = start - rng.integers(0, px_radius*2 * pars['shape_pars']['r'], 2)[None,...]
            idx = np.concatenate([start,end])
        idx.sort(axis=0)
        rotation = rng.integers(0,pars['shape_pars']['cardinality'])
        
        idx = rotate_point(idx,centroid,360*rotation/pars['shape_pars']['cardinality']).astype(int)
        idx.sort(axis=0)
        squares.append(idx)
    
    arr = np.zeros((pars['shape_pars']['shape'],pars['shape_pars']['shape']),dtype = bool)
    for idx in squares:
        arr[idx[0,0]:idx[1,0]+1,idx[0,1]:idx[1,1]+1] = True
    return arr

def rotate_point(v, origin, angle):
    angle = angle * np.pi / 180.0
    x = np.cos(angle) * (v[...,0]-origin) - np.sin(angle) * (v[...,1]-origin) + origin
    y = np.sin(angle) * (v[...,0]-origin) + np.cos(angle) * (v[...,1]-origin) + origin
    return np.array([x,y]).T

def min_thickness_test(arr,min_pixels):
    """
    Test for minimum connecting thickness within continuous features.
    Detects if any holes open up in feature.

    Args:
        arr (2D numpy array) : Binary-coercable array
        min_pixels (int)     : Minimum feature radius (1/2 total thickness) to test for

    Returns:
        Boolean
    """
    dil_arr = binary_erosion(arr,np.ones((min_pixels,min_pixels)))
    return label(np.invert(dil_arr.astype(bool))).max() == label(np.invert(arr.astype(bool))).max()

def min_support_test(arr,min_pixels):
    """
    Test for minimum connecting thickness within continuous features.
    Detects if there's a change in number of discrete features after dilation based on minimum thickness.

    Args:
        arr (2D numpy array) : Binary-coercable array
        min_pixels (int)     : Minimum feature radius (1/2 total thickness) to test for

    Returns:
        Boolean
    """
    dil_arr = binary_erosion(arr,np.ones((min_pixels,min_pixels)))
    return label(dil_arr.astype(bool)).max() == 1