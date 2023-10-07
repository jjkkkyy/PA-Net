import os
import h5py
import numpy as np
import random
import warnings
from shutil import copy
warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.realpath('__file__'))

def load_H5(directory, need_wavelength=False, keep_dtype = None):
    if need_wavelength is False:
        with h5py.File(directory, 'r') as f:
            image = f['Cube']['Images'][()]
        if keep_dtype is None:
            image = image.astype(np.float64)
        return image
    else:
        with h5py.File(directory, 'r') as f:
            image = f['Cube']['Images'][()]
            wavelength = f['Cube']['Wavelength'][()]
        if keep_dtype is None:
            image = image.astype(np.float64)
            wavelength = wavelength.astype(np.float64)
        return image, wavelength
    
def write_H5(filename, data, wavelength_data=None, original_dir=None):
    if original_dir is not None:
#         print('Transferring data')
        copy(original_dir, filename)
        hf = h5py.File(filename, 'r+')
        del hf['Cube']['Images']
        hf['Cube'].create_dataset('Images', data=data.astype(np.float32))
        hf.close()
    else:
        hf = h5py.File(filename, 'w')
        g1 = hf.create_group('Cube')
        g1.create_dataset('Images', data=data.astype(np.float32))
        if wavelength_data is not None:
            g1.create_dataset('Wavelength', data=wavelength_data)
        hf.close()
    return None

def load_H5_CT(directory):
    with h5py.File(directory, 'r') as f:
        image = f['kspace'].value
    return image

def normalization(data, data_max=None, data_min=None, rang=[-1,1], dynamic = None, std = False):  # normalize image to [-1,1]
    # dymanic (max) can be a number from 0-1
    if not std:
        if dynamic is None:
            if data_max is None:
                data_max = np.max(data)
            if data_min is None:
                data_min = np.min(data)
        else:
            data_max = np.quantile(data, dynamic)
            data_min = np.quantile(data, 1-dynamic)
        data[data>data_max]=data_max
        data[data<data_min]=data_min
        data = (rang[1]-rang[0])*(data-data_min)/(data_max-data_min)+rang[0]
        return data, data_max, data_min
    else:
        data_mean, data_std = data.mean(),data.std()
        data = (data-data_mean)/data_std
        data_max = np.max(data)
        data_min = np.min(data)
        if data_max - data.mean() >= data.mean() - data_min:
            data_min = data_max - 2*(data_max - data.mean())
        else:
            data_max = data_min + 2*(data.mean() - data_min)
        data = (rang[1]-rang[0])*(data-data_min)/(data_max-data_min)+rang[0]
        return data, data_max, data_min, data_mean, data_std

def redo_normalization(data, data_max=1, data_min=-1, data_mean=None, data_std=None, rang=[-1,1], std = False):
    if std:
        data = (data-data_mean)/data_std
    data[data>data_max]=data_max
    data[data<data_min]=data_min
    data = (rang[1]-rang[0])*(data-data_min)/(data_max-data_min)+rang[0]
    return data

def revert_normalization(data_norm, data_max=1, data_min=-1, data_mean=None, data_std=None, rang=[-1,1], std = False):
    data = (data_norm-rang[0])*(data_max-data_min)/(rang[1]-rang[0])+data_min
    if std:
        data = data * data_std + data_mean
    return data