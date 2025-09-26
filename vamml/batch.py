import os
import json
import warnings
from time import time

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from alive_progress import alive_bar
import pandas as pd

from .batchtools import shapes,images,alignments
from .batchtools.utils import NpEncoder,convert_keys_to_int
from .batchtools.default_pars import default_pars


# TODO Add all-in-one functions
class Experiment:
    def __init__(self,
                 base_dir,
                 expt_tag: str = None,
                 template: str = None,
                 cal: str|float = None,
                 random_seed: int = None,
                ):
        """ Generator and processor for VAMML experiment.

        Args:
            - base_dir (str) : Experiment directory. Creates if not existent.
            - expt_tag (str) : Sets experiment tag for parameters file (record-keeping).
                               If None, sets experiment name to last level of base_dir. Default: None
            - template (str)  : Loads template parameters from specified file location. See Experiment.save_template method. 
                                If str, tries to update defeult parameters from specified json file. 
                                If None, loads from default parameters. Default: None
            - cal (str,float) : Calibration value for images in mm/px or unit of voxel_size. 
                                If str, tries to load image to process for grid distance, if not found uses as cal_keyword
                                If not provided, tries to load from img_dir based on inclusion of self.pars['cal_keyword'] in name.
            - random_seed (int) : Sets RNG for generator, auto-generates if not provided. Default: None
        """
        # Parsing dirs
        self.base_dir = os.path.normpath(base_dir)
        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)
        self.img_dir = os.path.normpath(os.path.join(self.base_dir,'images'))
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir)
        # Loading parameters, either from local pars file or default
        try_pars = [i for i in os.listdir(self.base_dir) if 'pars.json' in i]
        if len(try_pars)==1:
            with open(os.path.join(self.base_dir,try_pars[0]),'r') as f:
                self.pars = json.load(f, object_hook = convert_keys_to_int)
        elif len(try_pars)>1:
            raise RuntimeError('Too many pars.json files found in base_dir.')
        else:
            self.pars = default_pars
        # expt_tag handling
        if (expt_tag is None) and ('expt' not in self.pars.keys()):
            self.pars['expt'] = [i for i in self.base_dir.split(os.path.sep) if i != ''][-1]
        elif type(expt_tag) == str:
            self.pars['expt'] = expt_tag
        elif expt_tag is not None:
            raise TypeError(f'Invalid expt_tag type: {type(expt_tag)}')
        # Initializing placeholder values
        self.pars['current_idx'] = 0
        if 'metadata' not in self.pars.keys():
            self.pars['metadata'] = {}
        if 'cal_keyword' not in self.pars.keys():
            self.pars['cal_keyword'] = 'cal'
        # Applying template if defined.
        if type(template) == str:
            try:
                with open(os.path.join(template),'r') as f:
                    self.update(json.load(f, object_hook = convert_keys_to_int))
            except:
                raise ValueError(f'Unable to parse template file as json: {template}')
        elif template is not None:
            raise TypeError(f'Invalid expt_tag type: {type(expt_tag)}')
        # cal handling
        if type(cal) == str:
            if os.path.isfile(cal):
                self.get_calibration(cal)
            else:
                self.pars['cal_keyword'] = cal
        elif cal is not None:
            self.get_calibration(cal)
        else:
            self.get_calibration()
        # random_seed handling
        if (random_seed is None) & (self.pars['random_seed'] is None):
            self.pars['random_seed'] = int(time())
        elif type(random_seed) == int:
            self.pars['random_seed'] = int(time())
        elif type(self.pars['random_seed']) is not int:
            raise TypeError(f'Invalid random_seed type: {type(random_seed)}')
        # Deprecating old nomenclature
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
        # Loading prepared base shapes if available. 
        self.base_shapes = []
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
        # Initializing function maps
        self.__shape_fcn_map = {
            'circle_deform':shapes.circle_deform,
            'rectangle_array': shapes.rectangle_array,
            'mixed':self.__mixed_shapes,
        }
        self.__img_fcn_map = {
            'coomassie':images.process_coomassie,
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

    def save_template(self,dst: str, pars_name: str = None):
        """
        Saves designated template parameters to designated (local) directory

        Args:
            dst (str)       : Directory to save parameters, with respect to current working directory.
            pars_name (str) : (Optional) Filename to write. Will be written as json filetype regardless.
                              Default saves as '{expt_tag} pars.json'  
        """
        keys = self.pars['template_pars']
        # Sorting in case the json encoder gets picky.
        keys.sort() 
        template = {}
        for k in keys:
            template[k] = self.pars[k]
        if pars_name is None:
            pars_name = self.__add_tag('template pars.json',force=True)
        else:
            try:
                pars_name = '.'.join([os.path.splitext(pars_name)[0],'json'])
            except:
                raise TypeError('pars_name could not be coerced to string.')
        with open(os.path.join(dst, pars_name), 'w') as f:
            json.dump(obj = template, fp = f, sort_keys = True, indent = 4, cls = NpEncoder)
        
    
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
        self.__save_pars()
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

    def get_calibration(self,
                        cal: str|float|bool = None):
        """Get calibration scale (unit/px). 
        Defaults to finding calibration image in ./images folder by inclusion of self.pars['cal_keyword'] in name.
        Processes on assumption of black grid with mm spacing on bright background.
        
        Args:
            cal (str,float,bool) : If str, tries to load as image file to parse as calibration grid.
                                   If float-like, applies as value (unit/px).
                                   If True, re-runs autodetection for calibration image from cal_keyword.
                                   If None, auto-detects if not run before. Default:None 
        """  
        # Get valid images
        image_list = self.__get_valid_images(self.img_dir) 
        # Try loading as filename
        if type(cal) == str:
            try:
                self.pars['cal_size'] = images.find_grid_dist(cal)
                self.pars['cal_image'] = cal
            except:
                raise RuntimeError(f'Unable to process calibration image: {cal}')
        # Load in numeric value
        elif (type(cal) == int) or (type(cal) == float):
            self.pars['cal_size'] = cal
        # Tries to autodetect
        elif (any(self.pars['cal_keyword'] in s for s in image_list) and 
              (((cal is None) and (self.pars['cal_size'] is None)) or (cal is True))):
            candidates = [i for i in image_list if self.pars['cal_keyword'] in i]
            if len(candidates) == 1:
                cal_image = image_list.pop(image_list.index(candidates[0]))
                self.pars['cal_image'] = cal_image
                self.pars['cal_size'] = images.find_grid_dist(os.path.join(self.img_dir,cal_image))
            else:
                raise RuntimeError(f'Too many candidate calibration images found: {candidates}')
        # Providing a default value if all else fails
        elif (self.pars['cal_size'] is None) and (len(image_list)>0):
            self.pars['cal_size'] = self.pars['voxel_size']
            warnings.warn('No calibration image/value found. Calibration scale defaulting to voxel_size.', UserWarning)
        while np.any([self.pars['cal_keyword'] in i for i in image_list]):
            cal_image = image_list.pop(image_list.index([i for i in image_list if self.pars['cal_keyword'] in i][0]))
        image_list.sort()
        self.pars['image_dict'] = dict(zip(range(len(image_list)),image_list))
        self.__save_pars()

    def process_images(self, 
                       export: bool = False,
                       process_fcn = None):
        """Processes images into grayscale arrays highlighting the core feature.
        Can implement custom functions. See batchtools.images.process_images or wiki for spec details. 

        Args:
            export (bool)         : Exports processed features to image_features.npy. Default False
            process_fcn (function): Function for image processing. Default None (loads from pars)"""
        # Calling processing function from function map
        if process_fcn is None:
            process_fcn = self.__img_fcn_map[self.pars['image_processing']]
        # Using to load images and patch up missing cal values
        self.get_calibration()
        # Defining empty list for iterator
        self.image_features = []
        # Going through image_dict to append image_features
        with alive_bar(len(self.pars['image_dict']), title='Processing images') as bar: 
            for i in range(len(self.pars['image_dict'])):
                image_name = os.path.join(self.img_dir,self.pars['image_dict'][i])
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
        if export:
            try:
                np.save(os.path.join(os.path.join(self.base_dir,'arrays'),'image_features.npy'),
                        [i['array'] for i in self.image_features])
            except:
                raise RuntimeError('Unable to export processed images to arrays.')
        

    def fit_images(self, 
                   auto_exclude: bool = None):
        """Fits image featuers with base shapes by downsampled rigid registration."""
        if not hasattr(self,'image_fits'):
            if auto_exclude is None:
                auto_exclude = self.pars['auto_exclude']
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
                # TODO add auto-exclude
                fit_pars = self.__get_fit_pars(crop_in,crops_out)
                self.pars['image_fits'][i] = fit_pars # pyright: ignore[reportPossiblyUnboundVariable]
                if (fit_pars['sim_score'] < self.pars['min_similarity']) and auto_exclude:
                    # skip_images auto-updates best_fit to None
                    self.skip_images(i)
                    print(f'No fit found for {self.pars['image_dict'][i]}')
                bar()
        self.__save_pars()

    def shapes_to_voxels(self):
        if hasattr(self,'shape_arrs'):
            arrs = self.base_shapes
        else:
            arrs = self.get_shapes()
        obj_z_iter = int(self.pars['obj_thickness'] / self.pars['voxel_size'])
        z_pitch_iter = int(self.pars['z_pitch'] / self.pars['voxel_size'])
        z_offset_iter = int(self.pars['z_offset'] / self.pars['voxel_size'])
        arrstack = np.concatenate([np.zeros_like(arrs[0])[None,...]]*z_offset_iter + 
                                  [np.concatenate([arr[None,...]]*obj_z_iter + 
                                                  [np.zeros_like(arr)[None,...]]*z_pitch_iter,axis=0) for arr in arrs],
                                  axis=0)
        arrstack = (arrstack.astype(bool).astype(np.uint8)*255)
        # TODO Add output format fcn_dict
        arrstack.tofile(os.path.join(self.base_dir,self.__add_tag('voxels.dat')))

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
                if self.pars['image_fits'][i]['best_fit'] is not None:
                    crop_out = images.centered_crop(self.base_shapes[self.pars['image_fits'][i]['best_fit']])
                    crop_out = crop_out.resize((self.pars['array_size'],
                                                self.pars['array_size']),
                                                resample=Image.Resampling.LANCZOS)
                    crop_out = images.norm_to_uint8(np.array(crop_out))[...,None]
                else:
                    crop_out = np.zeros_like(crop_in)
                im = np.concatenate([crop_in, crop_out, np.zeros_like(crop_in)], axis=-1)
                ax.imshow(im)
                if export:
                    im = Image.fromarray(im,mode='RGB')
                    if not os.path.isdir(os.path.join(self.base_dir,'alignments')):
                        os.mkdir(os.path.join(self.base_dir,'alignments'))
                    im.save(os.path.join(os.path.join(self.base_dir,'alignments'),
                                         self.pars['image_dict'][i]),mode='RGB')
                ax.set_title(i)
            ax.axis('off')
        if save:
            plt.savefig(os.path.join(self.base_dir,self.__add_tag('alignments.png')), bbox_inches='tight')
        else:
            plt.plot()

    def correct_alignment(self, 
                          image_number: int, 
                          shape_fit: int = None, 
                          rotation: int|float = 0,
                          mirror: bool = False):
        """Manually corrects alignment of misaligned shapes.

        Args:
            image_number (int)   : Index of image in image_dict
            shape_fit (int)      : Index of base shape (see Experiment.plot_shapes)
                                   If None, adds image_number to skip_images. Default: None
            rotation (int|float) : Value to rotate in degrees. Default: 0
            mirror (bool)        : Whether to flip on x-axis. Default: False
        """
        # TODO unified crop_in/out function?
        # TODO add interactive widget??
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(2,2))
        arr_in = self.image_features[image_number]['array']
        if mirror:
            arr_in = np.flip(arr_in, axis=-1)
        crop_in = images.centered_crop(arr_in)
        crop_in = crop_in.resize((self.pars['array_size'],
                                  self.pars['array_size']),
                                  resample=Image.Resampling.LANCZOS)
        crop_in = crop_in.rotate(-rotation,Image.Resampling.BICUBIC)
        crop_in = images.norm_to_uint8(np.array(crop_in))[...,None]
        # Standard correction
        if shape_fit is not None:
            # Sanity checking shape_fit value
            if shape_fit in range(self.pars['batch_size']):
                crop_out = images.centered_crop(self.base_shapes[shape_fit])
                crop_out = crop_out.resize((self.pars['array_size'],
                                            self.pars['array_size']),
                                            resample=Image.Resampling.LANCZOS)
                crop_out = images.norm_to_uint8(np.array(crop_out))[...,None]
                # Removing from skip_images if present
                while shape_fit in self.pars['skip_images']:
                    self.pars['skip_images'].remove(shape_fit)
            else:
                raise ValueError('shape_fit value not in range of batch_size.')
        # Adding skip_images if None
        else: 
            crop_out = np.zeros_like(crop_in)
            self.skip_images(image_number)
        # Updating pars with new fit
        update = {
            'best_fit': shape_fit, 
            'rotation': rotation,
            'mirror': mirror,
            }
        self.pars['image_fits'][image_number].update(update) 
        self.__save_pars()
        im = np.concatenate([crop_in, crop_out, np.zeros_like(crop_in)], axis=-1)
        ax.imshow(im)
        ax.axis('off')
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
            plt.savefig(os.path.join(self.base_dir,self.__add_tag('shapes.png')), bbox_inches='tight')
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
        for image_number in images:
            self.pars['image_fits'][image_number]['best_fit'] = None
        if append:
            self.pars['skip_images'].extend(images)
        else:
            self.pars['skip_images'] = images
        # Restructuring for compactness
        l = list(np.unique(self.pars['skip_images']))
        l.sort()
        self.pars['skip_images'] = l
        self.__save_pars()

    def export_alignments(self, relative=False , threshold=150):
        """
        Exports pngs of aligned shapes from input images (red) and source shape (green) to directory.
        Exports to './outputs' if relative=False, or './outputs_relative' if relative=True

        Args:
            relative (bool) : Whether to transform gel feature relative to input array.
            threshold (int) : Threshold to round up to max value in range of 0 to 255.

        Outputs: 
            Images - Images with gel feature in red channel, aligned to input array in green channel.
        """
        # TODO: fix up relative vs absolute packaging stuff, make summary images instead of output folders
        # Parameter to block in internal of image
        TRUE_IMAGE_THRESHOLD = threshold
        if relative:
            out_dir = os.path.join(self.base_dir,'outputs_relative')
        else:
            out_dir = os.path.join(self.base_dir,'outputs')
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        to_process = [i for i in range(len(self.image_features)) if i not in self.pars['skip_images']]
        to_package = {}
        with alive_bar(len(to_process), title='Exporting data') as bar:
            for i in to_process:
                px_size = self.pars['array_size']
                actual_size = self.pars['voxel_size'] * px_size
                scaled_size = int(actual_size/self.pars['cal_size'])
                
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
                # Unpacking to re-pack to array pkl later. Doing after relative since that has a potential processing step.
                to_package[i] = np.array(img)[...,:2] 
                img.save(os.path.join(out_dir,self.pars['image_dict'][i]))
                bar()
        packaged = []
        # Getting filler for entries to skip
        zero_template = np.zeros_like(to_package[list(to_package.keys())[0]])
        for i in range(len(self.image_features)):
            if i in self.pars['skip_images']:
                packaged.append(zero_template)
            else:
                packaged.append(to_package[i])
        if relative:
            np.save(os.path.join(os.path.join(self.base_dir,'arrays'),'output_relative.npy'),
                    packaged)
        else:
            np.save(os.path.join(os.path.join(self.base_dir,'arrays'),'output.npy'),
                    packaged)

    def __get_valid_images(self, src: str):
        files = []
        for f in os.listdir(src):        
            try:
                with Image.open(os.path.join(src,f)) as img:
                    img.verify()
                    files.append(f)
            except (IOError, SyntaxError):
                continue
        return files 
    
    def __mixed_shapes(self, seed: int, current_idx: int, **kwargs):
        if current_idx % 2 == 0:
            return shapes.circle_deform(seed,**self.pars)
        else:
            return shapes.rectangle_array(seed,**self.pars)
        
    def __add_tag(self,s: str, force: bool = False) -> str:
        if (self.pars['include_expt'] or force) and (type(self.pars['expt']) == str):
            return ' '.join([self.pars['expt'],s])
        else:
            return s
    
    def __save_pars(self):
        """
        Saves experiment parameters as json file to base directory.
        """
        # Structuring metadata in pars before saving
        keys = self.pars['metadata_pars']
        # Sorting in case the json encoder gets picky.
        keys.sort() 
        metadata = {}
        for k in keys:
            metadata[k] = self.pars[k]
        z_heights = {}
        z_pitch = self.pars['z_pitch']
        z_offset = self.pars['z_offset']
        obj_thickness = self.pars['obj_thickness']
        # Getting z-position with respect to where it should be based on array parameters and median object thickness
        for k in self.pars['image_dict'].keys():
            z_heights[k] = z_offset + z_pitch*k + obj_thickness*(k+0.5)
        metadata['z_height'] = z_heights
        self.pars['metadata'].update(metadata)
        pars_name = self.__add_tag('pars.json')
        with open(os.path.join(self.base_dir, pars_name), 'w') as f:
            json.dump(obj = self.pars, fp = f, sort_keys = True, indent = 4, cls = NpEncoder)

    def __get_fit_pars(self,img,outs):
        im_fit = {}
        adiff = np.abs(outs - img[None,None,...]).sum(axis=(-2,-1))
        asim = 1. - adiff / img.shape[-1]**2
        bestshape = asim.max(axis=1).argmax()
        alignmax = asim[bestshape].argmax()
        im_fit['best_fit'] = bestshape
        im_fit['rotation'] = alignmax % 360
        im_fit['mirror'] = bool(alignmax // 360)
        im_fit['sim_score'] = asim[bestshape,alignmax]
        return im_fit
        

def collate_metadata(base_dir: str,
                     out_csv: str = None,
                     expt_key: str = None
                     ):
    """Collates metadata generated from vamml.batch.Experiment for output into a reference csv for tensorflow use.
    This works by walking through the directories in a tree below base_dir, finding any parameter files and 
    creating records from there. It can additionally take an expt_key csv file, which adds metadata using expt as a key. 
    
    Args:
        base_dir (str) : highest level directory in which saved experiments are stored.
        out_csv (str)  : Name of csv with local directory to save output csv (eg 'dir/expts.csv')
                         if not specified, defaults to base_dir/metadata.csv
        expt_key (str) : Optional csv file to act as a dictionary with 'expt' column as key.
    """
    if out_csv is None:
        out_csv = os.path.join(base_dir,'metadata.csv')
    if expt_key is not None:
        try:
            exptdicts = pd.read_csv(expt_key).to_dict()
            exptidx = {v:k for k,v in exptdicts.pop('expt').items()}
        except:
            raise ValueError(f'Could not load expt_key with file {expt_key}')
    else: exptidx = {}
    records = []
    for d,_,fs in os.walk(base_dir):
        try_pars = [f for f in fs if ('pars.json' in f) and ('template' not in f)]
        # Only counting directories with one set of parameters
        if (len(try_pars)==1) and os.path.exists(os.path.join(d,'arrays')):
            with open(os.path.join(d,try_pars[0]),'r') as f:
                pars = json.load(f, object_hook = convert_keys_to_int)
                meta_pars = pars['metadata']
                if 'expt' in meta_pars.keys():
                    expt = meta_pars['expt']
                else:
                    expt = None
                meta_dicts = {}
                # Separating dictionaries from singleton values
                for k in [i for i in meta_pars.keys()]:
                    if type(meta_pars[k])==dict:
                        meta_dicts[k] = meta_pars.pop(k)
                # Iterating through image indices
                for i in range(pars['batch_size']):
                    if i not in pars['skip_images']:
                        # Setting up records for all output array sets. Also only outputting those that have complete arrays.
                        for arr in [os.path.join('arrays',i) for i in os.listdir(os.path.join(d,'arrays')) if 'output' in i]:
                            r = {}
                            r.update(meta_pars)
                            r['base_dir'] = d
                            r['source_array'] = arr
                            r['expt_index'] = i
                            r['base_image'] = os.path.join('images',pars['image_dict'][i])
                            # Reading individual index specific info
                            for k in meta_dicts.keys():
                                r[k] = meta_dicts[k][i]
                            # Adding info from expt_key file
                            if (expt is not None) and (expt in exptidx.keys()):
                                idx = exptidx[expt]
                                for k in exptdicts.keys():
                                    r[k] = exptdicts[k][idx]
                            records.append(r)
    # Compiling and saving dataframe to csv. Enforcing filetype.
    pd.DataFrame.from_records(records).to_csv('.'.join([os.path.splitext(out_csv)[0],'csv']),index=False)
              

    ### General outline:
    # Walk directory tree
    # Scan any directories with pars
    # Construct records from pars
    ### To include: metadata, dicts in metadata (from idx keys) src img (full local tree), src array + idx
    # Assemble dataframe from records