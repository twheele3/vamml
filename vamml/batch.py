import os
import json
from time import time

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from alive_progress import alive_bar

from .batchtools import shapes,images,alignments
from .batchtools.utils import NpEncoder


default_pars = {
                    "array_size": 256, # Int, array output size
                    "batch_size": 8, # Total number of features to generate.
                    "cal_keyword": "cal", # Keyword to locate calibration image from filename
                    "downsample_size": 64, # Size to downsample arrays to for difference testing
                    "hole_count": 2, # Max number of holes per feature.
                    "hole_margin": 0, # Margin to make holes inside main shape.
                    "hole_pars": { # Misc generator parameters, how to generate holes in feature
                        "n": 4,
                        "r": 0.4
                    },
                    "hole_range": [ # Range in pixels to randomize base hole diameter
                        10,
                        60
                    ],
                    "kmean_group": [ # For auto-fitting image features, nearest kmean group by HSV value.
                        155.40568757,
                        153.75507468,
                        111.35401764
                    ],
                    "mask_shape": "circle", # Masking element to clip vs object radius
                    "min_diff": 0.2, # Minimum fractional difference between different features.
                    "min_support": 0.5, # Minimum support of narrow points on feature, prevents fragmenting.
                    "min_thickness": 0.1, # Minimum thickness in mm between any internal features and surface.
                    "obj_radius": 3, # Maximum object radius
                    "obj_thickness": 1.25, # Z-thickness
                    "processing_function": "images.process_image", # (str) Name of function to call for imag processing
                    "random_seed": None, # Starting seed for generator
                    "shape_method": "circle_deform", # Method to generate shapes. Available: ['circle_deform', 'rectangle_array','mixed']
                    "shape_pars": { # Miscellaneous generator parameters to assemble shape arrays
                        "n": 6,
                        "r": 0.7,
                        "cardinality":4
                    },
                    "skip_images": [], # List of images to skip, by number key (see 'image_dict' when generated)
                    "tophat": False, # Whether to use tophat method for image processing. Not very stable?
                    "voxel_size": 0.025, # units/px 
                    "working_diameter": 5.0, # Maximum feature diameter from array center.
                    "z_offset":0.025, # Initial gap from base of voxel array to first feature.
                    "z_pitch": 0.625, # Distance between features in final voxel array.
                }





class Experiment:
    def __init__(self,
                 base_dir,
                 expt_tag = True,
                 img_dir = 'images',
                 pars = None,
                 random_seed = None,
                 cal = True,
                ):
        """ Generator and processor for VAMML experiment.

        Args:
            - base_dir (str) : Experiment directory. Creates if not existent.
            - expt_tag (str,bool) : Sets experiment tag for parameters file (record-keeping).
                                If bool==True, sets experiment name to last level of base_dir. Default: True
            - img_dir (str) : Sets image directory for images of printed gels. Creates if not existent. 
                              Must be subdirectory of base_dir. Default: 'images'
            - pars (bool,str) : Loads template parameters from specified file location. 
                                If None, loads from default parameters.
                                If True, tries to load from base_dir.
                                If str, tries to load from specified json file. 
                                Default: None
            - random_seed (int) : Sets RNG for generator, auto-generates if not provided. Default: None
            - cal (str,float) : Calibration value for images in mm/px or unit of voxel_size. 
                                If str, tries to load image to process for grid distance.
                                If not provided, tries to load from img_dir based on inclusion of self.pars['cal_keyword'] in name.
        
        """
        ### Parsing dirs
        self.base_dir = os.path.normpath(base_dir)
        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)
        self.img_dir = os.path.join(self.base_dir,img_dir)
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir)
        ### Parameter loading and handling. 
        # Loading from default
        if pars == None:
            self.pars = default_pars
        # Loading frome existing
        elif pars == True:
            # Loads from first thing fitting 'pars.json' template in base_dir.
            try_pars = [i for i in os.listdir(self.base_dir) if 'pars.json' in i]
            if len(try_pars)>0:
                with open(os.path.join(self.base_dir,try_pars[0]),'r') as f:
                    self.pars = json.load(f, object_hook = self.__convert_keys_to_int)
        # Loading from template
        else:
            with open(pars,'r') as f:
                self.pars = json.load(f, object_hook = self.__convert_keys_to_int)
        # Automatically setting cal_keyword to cal if not defined. 
        if 'cal_keyword' not in self.pars.keys():
            self.pars['cal_keyword'] = 'cal'
        ### Experiment tag handling
        # Interpreting tag from directory
        if expt_tag == True:
            self.pars['expt'] = [i for i in self.base_dir.split(os.path.sep) if i != ''][-1]            
        # Pulling from str
        elif type(expt_tag) == str:
            self.pars['expt'] = expt_tag
        elif (pars == True) & ('expt' in self.pars.keys()):
            pass
        else:
            self.pars['expt'] = None
        if (random_seed == None) & (self.pars['random_seed'] == None):
            self.pars['random_seed'] = int(time())
        # Initializing placeholder value
        self.pars['current_idx'] = 0

        ### Deprecating old nomenclature
        # 
        if 'image_fits' in self.pars.keys():
            for k in self.pars['image_fits'].keys():
                i = self.pars['image_fits'][k]
                if 'fit_score' in i.keys():
                    i['rotation'] = i['fit_score']
                    self.pars['image_fits'][k] = i
                    del self.pars['image_fits'][k]['fit_score']
        if 'shape' in self.pars['shape_pars'].keys():
            self.pars['array_size'] = self.pars['shape_pars']['shape']
            del self.pars['shape_pars']['shape']

        # Try to load in calibration if no previous value recorded. Otherwise defaults value.
        if 'image_cal' not in self.pars.keys():
            self.get_calibration()
        # Setting variable for general reference
        self.base_shapes = []
        # Load prepared base shapes by default. 
        if os.path.exists(os.path.join(base_dir,'arrays/base_shapes.npy')):
            self.from_array = True
            with open(os.path.join(base_dir,'arrays/base_shapes.npy'),'rb') as f:
                self.base_shapes = np.load(f)

        else:
            self.from_array = False
            
        # Load processed images by default.
        if os.path.exists(os.path.join(os.path.join(self.base_dir,'arrays'),'image_features.npy')):
            with open(os.path.join(os.path.join(self.base_dir,'arrays'),'image_features.npy'),'rb') as f:
                image_features = np.load(f)
            self.image_features = []
            if 'image_dict' not in self.pars.keys():
                self.pars['image_dict'] = {}
            for i,arr in enumerate(image_features):
                try:
                    imgname = self.pars['image_dict'][i]
                except:
                    # Generating names from enumeration for export use, if non-existent.
                    imgname = f'{i}.tif'
                    self.pars['image_dict'][i] = imgname
                self.image_features.append({'name':imgname, 'array':arr})
        else:
            self.image_features = []
        ### Initializing function maps
        self.__shape_fcn_map = {
            'circle_deform':shapes.circle_deform,
            'rectangle_array': shapes.rectangle_array,
            'mixed':self.__mixed_shapes,
        }
        self.__img_fcn_map = {
            'images.process_image':images.process_image,
        }
        
     

    def new_seed(self,seed = None):
        """
        Sets new random seed for shape generation.

        args:
            seed (int) (optional) : Random seed for numpy generator
        """
        if type(seed) != int:
            seed = int(time())
        self.pars['random_seed'] = seed
    
    def save_pars(self):
        """
        Saves experimental parameters as json file to base directory.
        """
        pars_name = ' '.join([str(i) for i in [self.pars['expt'],'pars.json'] if i != None])
        with open(os.path.join(self.base_dir, pars_name), 'w') as f:
            json.dump(obj = self.pars, fp = f, sort_keys = True, indent = 4, cls = NpEncoder)
        try:
            np.save(os.path.join(os.path.join(self.base_dir,'arrays'),'image_features.npy'),
                    [i['array'] for i in self.image_features])
        except:
            pass
    
    def update(self,pars):
        """Updates expt pars dictionary with supplied pars. Wrapper for self.pars.update(pars).
        
        Args:
            pars (dict): Dictionary of parameters to update"""
        self.pars.update(pars)


    def generate_shapes(self,batch_size=None,overwrite=False):
        """
        Generates new set of shapes based on parameters and random seed. 
        Shapes are validated for chance of cross-conformity by rigid registration comparisons.

        args:
            batch_size (int) : Number of shapes to generate. Default None (loads from pars)
            overwrite (bool)  : Whether to generate new seed when run. Default False.
        """
        if overwrite:
            self.new_seed()
        if batch_size == None:
            batch_size = self.pars['batch_size']
        else:
            self.pars['batch_size'] = batch_size
        rng = np.random.default_rng(self.pars['random_seed'])
        arrs = []
        seeds = []
        self.base_shapes = []
        # Minimum thickness in pixels for features to not have negative spaces connect
        self.pars['min_thickness_kernel'] = int(self.pars['min_thickness'] / 2 / self.pars['voxel_size'])
        self.pars['min_support_kernel'] = int(self.pars['min_support'] / 2 / self.pars['voxel_size'])
        # Aligns shapes by rigid registration, generating subsampled arrays to compare relative similarities.
        with alive_bar(1, title='Batch generated') as bar:
            crops = []
            # Generating preliminary candidates
            for i in range(batch_size):
                seed = rng.integers(0,2**32,1)[0]
                self.pars['current_idx'] = i
                arr = self.get_shape(seed)
                # Testing for shape continuity without any sections that are too thin
                while not np.all((
                    shapes.continuity_test(arr),
                    shapes.min_thickness_test(arr,**self.pars),
                    shapes.min_support_test(arr,**self.pars)
                )):
                    seed = rng.integers(0,2**32,1)[0]
                    arr = self.get_shape(seed)
                crop = images.centered_crop(arr)
                crop = crop.resize((self.pars['downsample_size'],
                                    self.pars['downsample_size']),
                                    resample=Image.Resampling.LANCZOS)
                arrs = np.array([np.array(crop.rotate(i,Image.Resampling.BICUBIC)) 
                                    for i in np.linspace(0.,360.-5,360//5)])
                arrs = np.concatenate([arrs,np.flip(arrs,axis=-1)],axis=0)
                crops.append(arrs)
                seeds.append(seed)
            crops = np.array(crops).astype(float)
            # Normalizing
            crops_out = (crops - crops.min()) / (crops.max() - crops.min())
            fits = np.ones((batch_size,batch_size),dtype=float)
            # Setting initial difference score matrix. Lower means more similar.
            for i in range(batch_size):
                    crop_in = crops_out[i][0]
                    scores = np.abs(crops_out - crop_in[None,None,...]).sum((2,3)) 
                    scores /= (crops_out[:,0,...] + crop_in[None,...]).sum((1,2))[...,None]
                    fits[i] = scores.min(1) - self.pars['min_diff']
            while np.any((fits < 0).sum(0)>1):
                
                if np.all(fits == 0):
                    break
                # Choosing best re-fit candidate by lowest difference score.
                i = (fits * (fits < 0)).sum(0).argmin()
                self.pars['current_idx'] = i
                fit_old = (fits[i] * (fits[i] < 0)).sum()
                fit_new = fit_old
                while fit_new <= fit_old:
                    seed = rng.integers(0,2**32,1)[0]
                    arr = self.get_shape(seed)
                    while not np.all((
                        shapes.continuity_test(arr),
                        shapes.min_thickness_test(arr,**self.pars),
                        shapes.min_support_test(arr,**self.pars)
                    )):
                        seed = rng.integers(0,2**32,1)[0]
                        arr = self.get_shape(seed)
                    crop = images.centered_crop(arr)
                    # Downsampling unit-cropped array for comparisons 
                    crop = crop.resize((self.pars['downsample_size'],
                                        self.pars['downsample_size']),
                                        resample=Image.Resampling.LANCZOS)
                    crop_in = np.array(crop)
                    crop_in = (crop_in - crop_in.min()) / (crop_in.max() - crop_in.min())
                    scores = np.abs(crops_out - crop_in[None,None,...]).sum((2,3)) 
                    scores /= (crops_out[:,0,...] + crop_in[None,...]).sum((1,2))[...,None]
                    scores = scores.min(1) - self.pars['min_diff']
                    scores[i] = 0 - self.pars['min_diff']
                    fit_new = (scores * (scores < 0)).sum()
                # Generating 3D array of rotations for 
                arrs = np.array([np.array(crop.rotate(i,Image.Resampling.BICUBIC))   # pyright: ignore[reportPossiblyUnboundVariable]
                                    for i in np.linspace(0.,360.-5,360//5)])
                arrs = np.concatenate([arrs,np.flip(arrs,axis=-1)],axis=0).astype(float)
                arrs = (arrs - arrs.min()) / (arrs.max() - arrs.min())
                crops_out[i] = arrs
                fits[i] = scores # pyright: ignore[reportPossiblyUnboundVariable]
                fits[:,i] = scores # pyright: ignore[reportPossiblyUnboundVariable]
                seeds[i] = seed # pyright: ignore[reportPossiblyUnboundVariable]
            bar()
        self.pars['shape_seeds'] = dict(zip(range(batch_size),seeds))
        self.save_pars()
        self.base_shapes = self.get_shapes()
        self.from_array = True
        if not os.path.exists(os.path.join(self.base_dir,'arrays')):
            os.mkdir(os.path.join(self.base_dir,'arrays'))
        np.save(os.path.join(os.path.join(self.base_dir,'arrays'),'base_shapes.npy'),self.base_shapes)

    def get_shape(self, seed, shape_fcn=None):
        if shape_fcn is None:
            try:
                shape_fcn = self.__shape_fcn_map[self.pars['shape_method']]
            except:
                raise ValueError("Invalid shape method.")
        return shape_fcn(seed,**self.pars)       
    
    def get_shapes(self):
        shape_list = []
        for i,v in enumerate(self.pars['shape_seeds'].values()):
            self.pars['current_idx'] = i 
            shape_list.append(self.get_shape(v))
        return shape_list

    def get_calibration(self, cal = True):
        """Get calibration scale (mm/px). 
        Defaults to finding calibration image in ./images folder by inclusion of self.pars['cal_keyword'] in name.
        Processes on assumption of black grid with mm spacing on bright background."""  
        # Get valid images
        image_list = self.__get_valid_images(self.img_dir) 
        # TODO remove if works [i for i in os.listdir(self.img_dir) if 'ipynb' not in i] 
        # Finding calibration image.
        if type(cal) == str:
            self.pars['image_cal'] = images.find_grid_dist(cal)
        elif any(self.pars['cal_keyword'] in s for s in image_list) & (cal == True):
            # Automatically process calibration image
            cal_image = image_list.pop(image_list.index([i for i in image_list if self.pars['cal_keyword'] in i][0]))
            self.pars['cal_image'] = cal_image
            self.pars['image_cal'] = images.find_grid_dist(os.path.join(self.img_dir,cal_image))
        elif (type(cal) == int) or (type(cal) == float):
            # Get input as specified number
            self.pars['image_cal'] = cal
        # Providing a default value for processing
        if 'image_cal' not in self.pars.keys():
            self.pars['image_cal'] = 1 / self.pars['voxel_size']
        while np.any([self.pars['cal_keyword'] in i for i in image_list]):
            cal_image = image_list.pop(image_list.index([i for i in image_list if self.pars['cal_keyword'] in i][0]))
        image_list.sort()
        self.pars['image_dict'] = dict(zip(range(len(image_list)),image_list))

    def process_images(self, process_fcn = None):
        """Processes images into grayscale arrays highlighting the core feature.
        Can implement custom functions. See batchtools.images.process_images or wiki for spec details. 

        Args:
            process_fcn (function): Function for image processing. Default None (loads from pars)"""
        # Function mapping processing function
        if process_fcn is None:
            process_fcn = self.__img_fcn_map[self.pars['processing_function']]
        # Loading images
        image_list = self.__get_valid_images(self.img_dir) 
        if 'image_cal' not in self.pars.keys():
            self.get_calibration()
        while np.any([self.pars['cal_keyword'] in i for i in image_list]):
            # TODO sort this out
            cal_image = image_list.pop(image_list.index([i for i in image_list if self.pars['cal_keyword'] in i][0]))
        image_list.sort()
        self.pars['image_dict'] = dict(zip(range(len(image_list)),image_list))
        self.image_features = []
        with alive_bar(len(image_list), title='Processing images') as bar: 
            for image_name in [os.path.join(self.img_dir,im) for im in image_list]:
                self.image_features.append(process_fcn(image_name, **self.pars))
                bar()
        shape_seeds = self.pars['shape_seeds'].values()
        # If not already defined, generating shapes from seeds
        if len(self.base_shapes) == 0:
            with alive_bar(len(shape_seeds), title='Processing shapes') as bar:
                if self.from_array:
                    for arr in self.base_shapes:
                        bar()
                else:
                    arrs = self.get_shapes() 
                    for arr in arrs:
                        self.base_shapes.append( arr.astype(int))
                        bar()

    def fit_images(self):
        if not hasattr(self,'image_fits'):
            self.pars['image_fits'] = {}
            crops = []
            # Generating downsampled comparison arrays
            for arr in self.base_shapes:
                crop = images.centered_crop(arr)
                crop = crop.resize((self.pars['downsample_size'],
                                    self.pars['downsample_size']),
                                    resample=Image.Resampling.LANCZOS)
                arrs = np.array([np.array(crop.rotate(i,Image.Resampling.BICUBIC)) 
                                    for i in np.linspace(0.,360.-1,360//1)])
                arrs = np.concatenate([arrs,np.flip(arrs,axis=-1)],axis=0)
                crops.append(arrs)
            crops = np.array(crops).astype(float)
            # Normalizing comparison maps
            crops_out = (crops - crops.min()) / (crops.max() - crops.min())
        with alive_bar(len(self.image_features), title='Aligning images') as bar:
            for i in range(len(self.image_features)):
                crop_in = images.centered_crop(self.image_features[i]['array'])
                crop_in = np.array(crop_in.resize((self.pars['downsample_size'],
                                                   self.pars['downsample_size']),
                                                   resample=Image.Resampling.LANCZOS))
                crop_in = crop_in.astype(float)
                # Normalizing comparison maps
                crop_in = (crop_in - crop_in.min()) / (crop_in.max() - crop_in.min())
                self.pars['image_fits'][i] = images.get_fit_pars(crop_in,crops_out) # pyright: ignore[reportPossiblyUnboundVariable]
                bar()

    def shapes_to_voxels(self):
        if hasattr(self,'shape_arrs'):
            arrs = self.base_shapes
        else:
            arrs = self.get_shapes()
        xy_size = int(self.pars['array_size'])
        obj_z_iter = int(self.pars['obj_thickness'] / self.pars['voxel_size'])
        z_pitch_iter = int(self.pars['z_pitch'] / self.pars['voxel_size'])
        z_offset_iter = int(self.pars['z_offset'] / self.pars['voxel_size'])
        arrstack = np.concatenate([np.zeros_like(arrs[0])[None,...]]*z_offset_iter + 
                                  [np.concatenate([arr[None,...]]*obj_z_iter + 
                                                  [np.zeros_like(arr)[None,...]]*z_pitch_iter,axis=0) for arr in arrs],
                                  axis=0)
        arrstack = (arrstack.astype(bool).astype(np.uint8)*255)
        arrstack.tofile(os.path.join(self.base_dir,f'{self.pars["expt"]} voxels.dat'))

    def plot_alignments(self, save = False, export = False):
        """
        Plots pointclouds of generated shapes (green) overlaid with processed image ROIs (red) to prove correct alignment.

        args:
            save (bool)       : Saves alignment figure in base_dir if True. Default False.
            export (bool)     : Exports individual alignments to base_dir/alignments if True. Default False.
        """
        rows = int(np.ceil(len(self.image_features)/4))
        fig,axs = plt.subplots(nrows=rows,ncols=4,figsize=(8,2*rows))
        for i in range(4*rows):
            if rows > 1:
                ax = axs[i//4][i%4]
            else:
                ax = axs[i]
            if i < len(self.pars['image_fits']):
                arr_in = self.image_features[i]['array']
                if self.pars['image_fits'][i]['mirror']:
                    arr_in = np.flip(arr_in, axis=-1)
                crop_in = images.centered_crop(arr_in)
                crop_in = crop_in.resize((self.pars['array_size'],
                                          self.pars['array_size']),
                                          resample=Image.Resampling.LANCZOS)
                crop_in = crop_in.rotate(-self.pars['image_fits'][i]['rotation'],Image.Resampling.BICUBIC)
                crop_in = images.norm_to_uint8(np.array(crop_in))[...,None]
                crop_out = images.centered_crop(self.base_shapes[self.pars['image_fits'][i]['best_fit']])
                crop_out = crop_out.resize((self.pars['array_size'],
                                            self.pars['array_size']),
                                            resample=Image.Resampling.LANCZOS)
                crop_out = images.norm_to_uint8(np.array(crop_out))
                im = np.concatenate([crop_in, crop_out[...,None], np.zeros_like(crop_in)], axis=-1)
                ax.imshow(im)
                if export:
                    im = Image.fromarray(im,mode='RGB')
                    if not os.path.isdir(os.path.join(self.base_dir,'alignments')):
                        os.mkdir(os.path.join(self.base_dir,'alignments'))
                    im.save(os.path.join(os.path.join(self.base_dir,'alignments'),self.pars['image_dict'][i]),mode='RGB')
                ax.set_title(i)
            ax.axis('off')
        if save:
            plt.savefig(os.path.join(self.base_dir,f'{self.pars['expt']} alignments.png'), bbox_inches='tight')
        else:
            plt.plot()

    def correct_alignment(self, image_number, shape_fit, rotation, mirror=False):
        """Manually corrects alignment of misaligned shapes.

        """
        self.pars['image_fits'][image_number] = {
            'best_fit': shape_fit, 
            'rotation': rotation,
            'mirror': mirror,
            }
        fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(2,2))
        arr_in = self.image_features[image_number]['array']
        if self.pars['image_fits'][image_number]['mirror']:
            arr_in = np.flip(arr_in, axis=-1)
        crop_in = images.centered_crop(arr_in)
        crop_in = crop_in.resize((self.pars['downsample_size']*4,self.pars['downsample_size']*4),resample=Image.Resampling.LANCZOS)
        crop_in = crop_in.rotate(-self.pars['image_fits'][image_number]['rotation'],Image.Resampling.BICUBIC)
        crop_in = images.norm_to_uint8(np.array(crop_in))[...,None]
        crop_out = images.centered_crop(self.base_shapes[self.pars['image_fits'][image_number]['best_fit']])
        crop_out = crop_out.resize((self.pars['downsample_size']*4,self.pars['downsample_size']*4),resample=Image.Resampling.LANCZOS)
        crop_out = images.norm_to_uint8(np.array(crop_out))
        im = np.concatenate([crop_in, crop_out[...,None], np.zeros_like(crop_in)], axis=-1)
        axs.imshow(im)
        axs.axis('off')
        plt.plot()

    def plot_shapes(self, save = False):
        if self.from_array:
            arrs = self.base_shapes
        else:
            arrs = self.get_shapes()

        rows = int(np.ceil(len(arrs)/4))
        fig,axs = plt.subplots(nrows=rows,ncols=4,figsize=(8,2*rows))
        for i in range(4*rows):
            if rows > 1:
                ax = axs[i//4][i%4]
            else:
                ax = axs[i]
            if i < len(arrs):
                ax.imshow(arrs[i],cmap='gray')
            ax.set_title(i)
            ax.axis('off')
        if save:
            plt.savefig(os.path.join(self.base_dir,f'{self.pars['expt']} shapes.png'), bbox_inches='tight')
        else:
            plt.plot()

    def skip_images(self,images=[], append=True):
        """
        Defines image numbers to skip when exporting outputs.

        args:
            images (list) : A list of integers corresponding to image numbers as shown by plot_alignments.
        """
        if not 'skip_images' in self.pars.keys():
            self.pars['skip_images'] = []
        if type(images) != list:
            images = [images]
        if append:
            for i in images:   
                self.pars['skip_images'].append(i)
        else:
            self.pars['skip_images'] = images

    def export_alignments(self, relative=False, threshold=150):
        """
        Exports pngs of aligned shapes from input images (red) and source shape (green) to directory.
        Exports to './outputs' if relative=False, or './outputs_relative' if relative=True

        Args:
            relative (bool) : Whether to transform gel feature relative to input array.
            threshold (int) : Threshold to round up to max value.

        Outputs: 
            Images - Images with gel feature in red channel, aligned to input array in green channel.
        """
        # Parameter to block in internal of image
        TRUE_IMAGE_THRESHOLD = threshold
        if relative:
            out_dir = os.path.join(self.base_dir,'outputs_relative')
        else:
            out_dir = os.path.join(self.base_dir,'outputs')
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        to_process = [i for i in range(len(self.image_features)) if i not in self.pars['skip_images']]
        with alive_bar(len(to_process), title='Exporting data') as bar:
            for i in to_process:
                px_size = self.pars['array_size']
                actual_size = self.pars['voxel_size'] * px_size
                scaled_size = int(self.pars['image_cal'] * actual_size)
                
                empty_in = np.zeros((scaled_size,scaled_size),dtype=float)
                empty_out = np.zeros((px_size,px_size),dtype=float)
                
                arr_in = self.image_features[i]['array']
                arr_out = self.base_shapes[self.pars['image_fits'][i]['best_fit']]
                
                if self.pars['image_fits'][i]['mirror']:
                    arr_in = np.flip(arr_in, axis=-1)
                crop_in = images.centered_crop(arr_in)
                crop_in = crop_in.rotate(-self.pars['image_fits'][i]['rotation'],Image.Resampling.BICUBIC)
                crop_in = np.array(crop_in).astype(float)
                in_offset = (np.asarray(empty_in.shape) - np.asarray(crop_in.shape))//2
                empty_in[in_offset[0]:in_offset[0]+crop_in.shape[0],in_offset[1]:in_offset[1]+crop_in.shape[1]] = crop_in
                in_scaled = np.array(Image.fromarray(empty_in).resize((px_size,px_size)))
                # Cleaning up floating point discrepancies
                in_scaled[in_scaled<0] = 0
                #### Rescaling to fill in bulk of shape based on histogram analysis.
                in_scaled = images.norm_to_uint8(in_scaled).astype(float)
                ### Commented out for new method
                in_scaled = in_scaled / TRUE_IMAGE_THRESHOLD

                in_scaled[in_scaled>1.] = 1.
                in_scaled = images.norm_to_uint8(in_scaled)
                
                crop_out = images.centered_crop(arr_out)
                # Patch to fix bug created after trying to fix centered_crop 2024-10-10
                if np.any((np.array(empty_out.shape) - np.array(crop_out.size))<0):
                    crop_out = crop_out.resize(arr_out.shape)
                crop_out = np.array(crop_out).astype(float)
                # print(f'arr:{arr_out.shape},crop_out:{crop_out.shape},empty:{empty_out.shape}')
                out_offset = (np.asarray(empty_out.shape) - np.asarray(crop_out.shape))//2
                empty_out[out_offset[0]:out_offset[0]+crop_out.shape[0],out_offset[1]:out_offset[1]+crop_out.shape[1]] = crop_out
                
                image = np.concatenate([in_scaled[...,None],
                                        images.norm_to_uint8(empty_out)[...,None],
                                        np.zeros_like(empty_out).astype(np.uint8)[...,None]],axis=-1)
                img = Image.fromarray(image)
                if relative:
                    img = alignments.align_affine_trs(img)
                img.save(os.path.join(out_dir,self.pars['image_dict'][i]))
                bar()

    def __get_valid_images(self,src: str):
        files = []
        for f in os.listdir(src):        
            try:
                with Image.open(os.path.join(src,f)) as img:
                    img.verify()
                    files.append(f)
            except (IOError, SyntaxError):
                continue
        return files
        
    def __convert_keys_to_int(self,d: dict):
        new_dict = {}
        for k, v in d.items():
            try:
                new_key = int(k)
            except ValueError:
                new_key = k
            if type(v) == dict:
                v = self.__convert_keys_to_int(v)
            new_dict[new_key] = v
        return new_dict     
    
    def __mixed_shapes(self, seed, current_idx, **kwargs):
        if current_idx % 2 == 0:
            return shapes.circle_deform(seed,**self.pars)
        else:
            return shapes.rectangle_array(seed,**self.pars)