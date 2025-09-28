import os
import json

import numpy as np

class NpEncoder(json.JSONEncoder):
    """Encodes python dicts with numpy formats to json readable mode"""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)
    
def convert_keys_to_int( d: dict) -> dict:
        """Recursively converts dict keys to integers. Useful for importing json files.
        """
        new_dict = {}
        for k, v in d.items():
            try:
                new_key = int(k)
            except ValueError:
                new_key = k
            if type(v) == dict:
                v = convert_keys_to_int(v)
            new_dict[new_key] = v
        return new_dict

def load_json(filename: str) -> dict:
    """Reads json file and returns dict with values."""
    with open(filename,'r') as f:
        return json.load(f, object_hook = convert_keys_to_int)
    
def save_json(filename: str, d: dict) -> None:
    """Saves dict as filename"""
    with open(filename, 'w') as f:
        json.dump(obj = d, fp = f, sort_keys = True, indent = 4, cls = NpEncoder)