default_pars = {
                    "array_size": 256, # Int, array output size
                    "auto_exclude": True,
                    "batch_size": 8, # Total number of features to generate.
                    "cal_keyword": "cal", # Keyword to locate calibration image from filename
                    "cal_size": None, # Calibration value in units/px, same as voxel_size 
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
                    "image_processing": "coomassie", # (str) Name of function to call for imag processing
                    "include_expt": False, # Adds expt tag to certain outputs 
                    "intermediate_scale": 2.0, # Amount to scale array_size up for saving image_features (preserves detail)
                    "mask_shape": "circle", # Masking element to clip vs object radius
                    "metadata" : {
                        "dose":"nan"
                    },
                    "metadata_pars": [ # Pars to add key:value pairs to metadata when saving
                        "expt",
                        "voxel_size"
                    ],
                    "min_diff": 0.2, # Minimum fractional difference between different features.
                    "min_similarity" : 0.8, # Minimum similarity threshold, discards fits below
                    "min_support": 0.5, # Minimum support of narrow points on feature, prevents fragmenting.
                    "min_thickness": 0.1, # Minimum thickness in mm between any internal features and surface.
                    "obj_radius": 3, # Maximum object radius
                    "obj_thickness": 1.25, # Z-thickness
                    "printer_format": "tomolite",
                    "random_seed": None, # Starting seed for generator
                    "shape_method": "circle_deform", # Method to generate shapes. Available: ['circle_deform', 'rectangle_array','mixed']
                    "shape_pars": { # Miscellaneous generator parameters to assemble shape arrays
                        "n": 6,
                        "r": 0.7,
                        "cardinality":4
                    },
                    "skip_images": [], # List of images to skip, by number key (see 'image_dict' when generated)
                    "template_pars": [
                        "array_size",
                        "auto_exclude",
                        "batch_size",
                        "cal_keyword",
                        "hole_count",
                        "hole_margin",
                        "downsample_size",
                        "hole_pars",
                        "hole_range",
                        "include_expt",
                        "mask_shape",
                        "metadata_pars",
                        "min_diff",
                        "min_similarity",
                        "min_support",
                        "min_thickness",
                        "obj_radius",
                        "obj_thickness",
                        "processing_function",
                        "shape_method",
                        "shape_pars",
                        "template_pars",
                        "tophat",
                        "voxel_size",
                        "working_diameter",
                        "z_offset",
                        "z_pitch"
                    ],
                    "tophat": False, # Whether to use tophat method for image processing. Not very stable?
                    "voxel_size": 0.025, # units/px 
                    "working_diameter": 5.0, # Maximum feature diameter from array center.
                    "z_offset":0.025, # Initial gap from base of voxel array to first feature.
                    "z_pitch": 0.625, # Distance between features in final voxel array.
                }