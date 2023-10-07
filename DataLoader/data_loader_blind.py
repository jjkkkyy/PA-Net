import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import json
import random
import joblib


class DataLoader():
    '''
    dataset_name:
        standard_{} int, default sigma
    '''

    def __init__(self, file_name, dataset_name, img_res=(128, 128), channel=32):
        self.file_name = file_name
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.noise_std = int(dataset_name.split('_')[1])
        self.channel = int(channel)
        self.noise_list = [0, 5, 10, 20, 50]
        self.dataset_dict = {}
        self.hr_img = None
        
        DATA_FOLDER = ''  # Set empty to enable default path

        script_dir = os.path.dirname(os.path.realpath('__file__'))
        path = Path(script_dir)

        if not DATA_FOLDER:
            DATA_FOLDER = os.path.join(str(path), 'Data', 'Train')
        
        for sigma in self.noise_list:
            with open(os.path.join(str(path), 'Data', 'Train', '{}_{}.joblib'.format(self.file_name, int(sigma))), 
                                   'rb') as handle:
                data = joblib.load(handle)
                if self.hr_img is None:
                    self.hr_img = {
                        'train': data['train']['HR'],
                        'valid': data['valid']['HR'],
                        'test': data['test']['HR'],
                    }
                self.dataset_dict[sigma] = data
                
        print(self.dataset_dict.keys(), 'loaded')
        
        
    def load_data(self, batch_size=1, data_type='train', sigma=None, random_noise=False):
        '''
        sigma: int or list of int
        
        random_noise True or not exists file -> random
        random_noise False and exits file -> from file
        '''
        if sigma is None:
            sigma = self.noise_std
        
        
        if isinstance(sigma, list):
            # It's a list
            if batch_size != len(sigma):
                raise Exception('sigma list must match batch size!')
        else:
            sigma = batch_size * [sigma]
        
        
        if data_type == 'train':
            
            img_res = self.img_res
            input_res = self.hr_img[data_type].shape
            imgs_hr = np.zeros((batch_size, img_res[0], img_res[1], self.channel))
            imgs_lr = np.zeros((batch_size, img_res[0], img_res[1], self.channel))
            c_max = int(input_res[2]-self.channel)
            
            for i in range(batch_size):
                noise_level = int(sigma[i])
                img_hr = self.hr_img[data_type]
                if random_noise or noise_level not in self.noise_list:
                    # Produce random noise
                    img_lr = img_hr + np.random.normal(0, noise_level/255*2, img_hr.shape)
                else:
                    # Load LR directly from memory
                    img_lr = self.dataset_dict[noise_level][data_type]['LR']
                
                random_x = np.random.randint(img_hr.shape[0]-img_res[0]+1, size=1)[0]
                random_y = np.random.randint(img_hr.shape[1]-img_res[1]+1, size=1)[0]
                img_hr = img_hr[random_x:random_x+img_res[0], random_y:random_y+img_res[1], :]
                img_lr = img_lr[random_x:random_x+img_res[0], random_y:random_y+img_res[1], :]
                
                if input_res[2] >= self.channel:
                    random_c = np.random.randint(-self.channel//2+1, high=input_res[2]-self.channel//2, size=1)[0]
                    if random_c < 0:
                        random_c = 0
                    elif random_c > c_max:
                        random_c = c_max
                    img_hr = img_hr[:,:,random_c:random_c+self.channel]
                    img_lr = img_lr[:,:,random_c:random_c+self.channel]
                else:
                    img_hr = np.concatenate((img_hr, np.dstack([img_hr[:,:,-1]]*int(self.channel-input_res[2]))), axis=-1)
                    img_lr = np.concatenate((img_lr, np.dstack([img_lr[:,:,-1]]*int(self.channel-input_res[2]))), axis=-1)
                
                # Clip noisy data
                img_hr = np.clip(img_hr, -1, 1)
                img_lr = np.clip(img_lr, -1, 1)
                
                if np.random.random() < 0.5:
                    img_hr = np.fliplr(img_hr)
                    img_lr = np.fliplr(img_lr)
                
                imgs_hr[i] = img_hr
                imgs_lr[i] = img_lr
                
        elif data_type == 'valid' or data_type == 'test':
            
            img_res = self.hr_img[data_type].shape
            
            imgs_hr = np.zeros((batch_size, img_res[0], img_res[1], img_res[2]))
            imgs_lr = np.zeros((batch_size, img_res[0], img_res[1], img_res[2]))
            
            for i in range(batch_size):
                noise_level = int(sigma[i])
                img_hr = self.hr_img[data_type]
                if random_noise or noise_level not in self.noise_list:
                    # Produce random noise
                    img_lr = img_hr + np.random.normal(0, noise_level/255*2, img_hr.shape)
                else:
                    # Load LR directly from memory
                    img_lr = self.dataset_dict[noise_level][data_type]['LR']
                
                # No need to sample valid & test
                # Clip noisy data
                img_hr = np.clip(img_hr, -1, 1)
                img_lr = np.clip(img_lr, -1, 1)
                img_hr = np.clip(img_hr, -1, 1)
                imgs_hr[i] = img_hr
                imgs_lr[i] = img_lr
                
        return imgs_hr, imgs_lr
