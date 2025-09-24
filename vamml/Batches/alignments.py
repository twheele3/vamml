import numpy as np
from PIL import Image
from scipy.optimize import minimize

def affine_trs_decomp(x,y,s,theta,cx,cy):
    """
    Affine (t)ranslation (r)otation (s)caling decomposition for individually integrating cartesian-parseable inputs and create a usable affine matrix output.

    Args:
        x (float)     : X-axis translation value
        y (float)     : Y-axis translation value
        s (float)     : Scale relative to base value of 1
        theta (float) : Angle to rotate in radians

    returns:
        array shape (6,) describing affine matrix per Image.Transform.AFFINE standard
    """
    mat_t = np.array([[1, 0, x],
                      [0, 1, y],
                      [0, 0, 1]],
                    dtype = float)
    mat_r = np.array([[np.cos(theta), -np.sin(theta), -cx*np.cos(theta) + cy*np.sin(theta) + cx],
                      [np.sin(theta),  np.cos(theta), -cx*np.sin(theta) - cy*np.cos(theta) + cy],
                      [0, 0, 1]],
                    dtype = float)
    mat_s = np.array([[1/s, 0, cx - cx/s],
                      [0, 1/s, cy - cy/s],
                      [0, 0, 1]],
                    dtype = float)
    res = mat_s @ mat_t @ mat_r
    affine_matrix = res.flatten()[:6]
    return affine_matrix

def opt_affine_trs(x0,img):
    """
    Optimizes an affine transform for translation, rotation, scale (trs) to align an image pair. Expects image pair to be provided as red and green channels of a PIL Image object. Green channel is used as target to align red. Rough pre-alignment is recommended.

    Args:
        img (PIL.Image object)   : An array with 2+ channels. Aligns red channel to green channel.
        x0 (list-like of floats) : A length-4 list-like of floats, for initial prediction. Each represent: [x_transform, y_transform, scale, degrees]

    Returns:
        float : a score of non-overlap. Minimized by optimal overlap.
    """
    x,y,s,theta = x0
    cx,cy = np.array(img.size[:2])/2
    affine_matrix = affine_trs_decomp(x,y,s,theta,cx,cy)
    red,green = img.split()[:2]
    red = red.transform(img.size, Image.Transform.AFFINE, affine_matrix, resample=Image.Resampling.BICUBIC)
    r = np.array(red).astype(int)
    g = np.array(green).astype(int)
    score = np.abs(r-g).sum()
    return score

def align_affine_trs(img,x0=[0.,0.,1.,0.]):
    """
    Optimizes an affine transform for translation, rotation, scale (trs) to align an image pair. Expects image pair to be provided as red and green channels of a PIL Image object. Green channel is used as target to align red. Rough pre-alignment is recommended.

    Args:
        img (PIL.Image object)   : An array with 2+ channels. Aligns red channel to green channel.
    
    Returns:
        PIL.Image object : Image with red channel optimally aligned to green channel.
    """
    res = minimize(opt_affine_trs, x0, args=(img),method='nelder-mead',
                                  options={'xatol': 1e-6, 'disp': False})
    red,green = img.split()[:2]
    cx,cy = np.array(img.size[:2])/2
    red = red.transform(img.size, Image.Transform.AFFINE, affine_trs_decomp(res.x[0],res.x[1],res.x[2],res.x[3],cx,cy), 
                        resample=Image.Resampling.BICUBIC)
    return Image.merge(img.mode,tuple([red,green] + list(img.split()[2:])))