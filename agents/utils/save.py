import json

import numpy as np


def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def save_np(arr, path):
    np.save(path, arr)