import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import distance_matrix
from skimage.filters import gaussian,sobel
from skimage.measure import label
from skimage.morphology import white_tophat,binary_dilation,binary_erosion
from sklearn.cluster import KMeans

from .shapes import circle_deform,rectangle_array

def find_grid_dist(image,plot=False):
    """
    Determines the pixel calibration distance between lines of a calibration grid.
    Optionally plots proof of function.

    args:
        im (arr[N,M]) : A 2D array of an image of a calibration grid
        plot (bool)   : Whether to plot proof in-line

    returns: 
        float : pixels per grid unit calibration distance
    """
    if type(image)==str:
        image = np.asarray(Image.open(image)).sum(axis=-1)
    im = gaussian(image, sigma=10)
    # filtering by dxdy sobel
    im = sobel(im,axis=0)*sobel(im,axis=1)
    # refiltering for magnitude
    im = sobel(im)
    # blurring 
    im = gaussian(im,sigma=10)
    im = im > (im.max() * 0.5)
    basins = label(im)
    pts = []
    for i in range(basins.max()):
        # incrementing for boolean test reasons
        i += 1
        # Find centroid of each dot
        pts.append( np.asarray(np.where(basins == i)).mean(axis=-1) )
    pts = np.asarray(pts)
    # Find nearest neighbors of each point
    dists = distance_matrix(pts,pts)
    dists_sorted = np.sort(dists,axis=-1)
    # Diagnostic plotting as proof
    if plot:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(image,cmap='gray')
        ax.scatter(pts[:,1],pts[:,0],color='red',s=5)
        plt.show()
    return np.median(dists_sorted[:,1:3].flatten())

def norm_arr(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def normalize_to_percentiles(arr,lower,upper):    
    # Casting to float
    arr = arr.astype(float)
    # Sanity checking values to make sure they fit in function
    lower = max(lower,0)
    upper = min(upper,1)

    # Getting quantile values
    lower_val = np.quantile(arr,lower)
    upper_val = np.quantile(arr,upper)
    
    # Normalizing to quantiles
    arr = (arr - lower_val) / (upper_val - lower_val)
    
    # Trimming values to [0,1] range
    arr[arr > 1] = 1
    arr[arr < 0] = 0
    return arr
    
def process_image(image_name,image_cal,tophat=False,**kwargs):
    """Processes images into boolean feature arrays based on HSV. 
    Presumes high saturation object on low saturation background.
    
    Args:
        image_name (str)    : Complete file path for image to process
        image_cal (float)   : Value for units/px. Presumes mm/pix for reasonable processing.
        tophat (bool,float) : Whether to use tophat background subtraction method. Bypasses if False, 
                              otherwise uses value as multiplier with image_cal. The subsequent value 
                              should be slightly larger than average radius of features to detect.     
    """
    hsv = np.asarray(Image.open(image_name).convert('HSV')).astype(float)
    hsv[...,2] = 255 - hsv[...,2]
    coeffs = [2,1,1]
    kmeans = KMeans(n_clusters = 2, n_init = 10)
    kmeans.fit(hsv.reshape((np.prod(hsv.shape[:-1]),hsv.shape[-1])))

    # Experimental tophat method for separating by hue intensity 
    if tophat is not False:
        mask = white_tophat(hsv[...,1:].prod(-1),
                            footprint=np.ones((int(image_cal * tophat),int(image_cal * tophat))))
    else:
        mask = hsv[...,2]
    mask = mask > (mask.max() * 0.5)
    # Creating mask to cover up small noisy features in image (fibers, particles, etc)
    mask = binary_dilation(binary_erosion(mask,np.ones((int(image_cal * 0.1),int(image_cal * 0.1)))),
                           np.ones((int(image_cal * 0.1),int(image_cal * 0.1))))
    # Weighting coefficients for channel significance
    coeffs = [2,1,1]
    arr = np.asarray([hsv[...,i]*coeffs[i] for i in range(3)]).sum(axis=0) * mask
    arr = (arr - arr.min())*255/(arr.max() - arr.min())
    arr = normalize_to_percentiles(arr,
                                   lower = 1 - 1.*mask.sum() / np.prod(mask.shape),
                                   upper = 1 - 0.05*mask.sum() / np.prod(mask.shape))
    basins = label(mask)
    n,c = np.unique(basins,return_counts=True)
    c = c[n!=0]
    n = n[n!=0]
    id = n[c.argmax()]
    obj = basins==id
    arr = obj*arr
    # Normalizing image array
    arr = arr.astype(float)
    arr[arr < 1e-6] = 0
    arr[arr > 1] = 1
    return {'name':os.path.split(image_name)[1],'array':arr}

def get_fit_pars(img,outs):
    im_fit = {}
    adiff = np.abs(outs - img[None,None,...]).sum(axis=(-2,-1))
    asim = 1. - adiff / img.shape[-1]**2
    bestshape = asim.max(axis=1).argmax()
    align1 = asim[bestshape].argmax()
    im_fit['best_fit'] = bestshape
    im_fit['rotation'] = align1 % 360
    im_fit['mirror'] = bool(align1 // 360)
    return im_fit

def centered_crop(arr):
    """
    args:
        arr (numeric, [N,M]): Array where higher values correlate to feature of interest. Assumes only one feature.
    returns:
        PIL.Image object corresponding to crop allowing for full range of rotation without cropping.

    TODO: Auto-pad image if bounding box exceeds image size.
    """
    arr = arr.astype(float)
    idx = np.array(np.where(arr > 0.1*(arr.max()-arr.min())))
    centroid = idx.mean(axis=1)
    scale = (((idx - centroid[...,None])**2).sum(axis=0)**0.5).max()
    # Pad array if necessary
    while np.any((centroid - scale) < 0) or np.any((centroid + scale) >= arr.shape):
        idx = np.array(np.where(arr > 0.1*(arr.max()-arr.min())))
        centroid = idx.mean(axis=1)
        scale = (((idx - centroid[...,None])**2).sum(axis=0)**0.5).max()
        pad_size = (np.array(arr.shape)*1.4).astype(int)
        pad_delta = (pad_size - arr.shape) // 2
        centroid = centroid + pad_delta
        pad_arr = np.zeros(pad_size).astype(arr.dtype)
        pad_arr[pad_delta[0]:pad_delta[0]+arr.shape[0],pad_delta[1]:pad_delta[1]+arr.shape[1]] = arr
        arr = pad_arr
    cropbox = np.concatenate([np.floor(centroid - scale).astype(int),np.ceil(centroid + scale).astype(int)])
    crop = arr[cropbox[0]:cropbox[2],cropbox[1]:cropbox[3]]
    crop = (255*(crop - crop.min()) / (crop.max() - crop.min())).astype(np.uint8)
    return Image.fromarray(crop,mode='L')

def norm_to_uint8(arr):
    return (255*(arr.astype(float) - arr.astype(float).min()) / (arr.astype(float).max() - arr.astype(float).min())).astype(np.uint8)
