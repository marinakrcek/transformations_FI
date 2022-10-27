import numpy as np
import json
import os
from enum import Enum
import pandas as pd


class fault_type(Enum):
    normal = 0
    reset = 1
    success = 2
    Pass = 3
    Fail = 4
    Mute = 5
    changing = 6


def read_config_file(root, file):
    """
    Function reads configuration file.
    Example is below. (JSON file)
    {
      "transformation": {"name": "polynomial" , "degree":2},
      "out_of_scope": "scale",
      "x": {"min":0, "max":24, "step":0.05, "transform": true},
      "y": {"min":0, "max":24, "step":0.05, "transform": true},
      "z": {"min":0, "max":5, "step":0.05, "transform": true},
      "time": {"min":365, "max":375, "step":5, "transform": true},
      "duration": {"min":1, "max":2, "step":0.5, "transform": true},
      "laser pulse width": {"min":0, "max":24, "step":0.05, "transform": false},
      "laser pulse height": {"min":0, "max":24, "step":0.05, "transform": false},
      "trigger delay": {"min":0, "max":24, "step":0.05, "transform": false},
      "laser intensity": {"allowed_values": [1100, 1200], "transform": false}
    }

    :param root: root folder where file is
    :param file: name of the file in the root folder with configuration
    :return: transformation dictionary with name of the transformation and degree value if polynomial;
             data is dictionary with names of parameters as keys, while value is dictionary with information
             about the parameter, such as min and max value, step, and should it be transformed,
             also it holds information on how the transformed values out of scope are handled (clip, modulo or scale);
    """
    with open(os.path.join(root, file), 'r') as fp:
        # dict with keys being name of parameters, and the value is a dict with some specifications
        data: dict = json.load(fp)
    transformation_info = data['transformation']
    del data['transformation']
    return transformation_info, data


def save_transformed(file, metadata, header, transformed):
    """
    for saving simulated transformed data
    add first invalid input
    """
    with open(file, 'w') as fp:
        json.dump({"metadata": metadata, "header": header.tolist(),
                   "results": [["0000", "0000", 0.0, 0.0, "0.0", "Pass", "0", "0"]] + transformed.tolist()}, fp,
                  indent=4)


def save_reversed_transformed(file, metadata, header, transformed):
    with open(file, 'w') as fp:
        json.dump({"metadata": metadata, "columns": header, "data": transformed.tolist()}, fp, indent=4)


def read_faults_file(root, file, groupby=True):
    """
    Reads a file of simulated data
    :param root: start/root folder from which all the files are read and saved to
    :param file: name of the file to read
    :return: metadata dictionary, header with name of the parameters, results (attack parameters and outputs) as np.array
    """
    with open(os.path.join(root, file), 'r') as fp:
        data = json.load(fp)
    metadata = data['metadata']
    param_bounds = metadata['parameter_bounds']
    param_names = list(param_bounds.keys())
    header = data['header']
    indexes = [header.index(name) for name in param_names]
    assert (len(indexes) == len(param_names))
    status = 'STATUS'
    indexes.append(header.index(status))
    results = np.array(data['results'][1:])  # first entry is not valid
    df = pd.DataFrame(results[:, indexes], columns=param_names + [status])
    if groupby:
        df = pd.DataFrame(
            df.groupby(param_names)[status].apply(
                lambda x: list(x)[0] if len(set(x)) == 1 else 'changing')).reset_index()
    return data['metadata'], np.array(param_names + [status]), np.array(df)


def get_filename(path):
    _, tail = os.path.split(path)
    return tail
