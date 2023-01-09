import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import xarray as xr

from datasets import register

@register('natl')
class NATL(Dataset):

    def __init__(self, root_path, first_k=None,
                 repeat=1, cache='bin'):
        self.repeat = repeat
        self.cache = cache

        self.files = []
        npz_data = np.load(root_path)
        n_var, n_time, n_lat, n_lon = np.shape(npz_data['FdataAllVar']) 
        npz_data = npz_data["FdataAllVar"][0,:,:,0:n_lat]

        for t in range(n_time):

            #if cache == 'none':
             #   self.files.append(file)

            if cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, str(t) + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(npz_data[t], f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    npz_data[t]))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        #if self.cache == 'none':
         #   return transforms.ToTensor()(Image.open(x).convert('RGB'))

        if self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = x[np.newaxis,...]
            x = torch.from_numpy(x).float() 
            return x

        elif self.cache == 'in_memory':
            return x

@register('natl-sst')
class NATLSST(Dataset):

    def __init__(self, root_path, first_k=None,
                 repeat=1, cache='bin'):
        self.repeat = repeat
        self.cache = cache

        self.files = []
        npz_data = np.load(root_path)
        n_var, n_time, n_lat, n_lon = np.shape(npz_data['FdataAllVar']) 
        ssh_data = npz_data["FdataAllVar"][0,:,:,0:n_lat]
        sst_data = npz_data["FdataAllVar"][1,:,:,0:n_lat]

        for t in range(n_time):

            data = np.stack((ssh_data[0], sst_data[0]))
            #if cache == 'none':
             #   self.files.append(file)

            if cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path)+'_sst_')
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, str(t) + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(data, f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    data))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        #if self.cache == 'none':
         #   return transforms.ToTensor()(Image.open(x).convert('RGB'))

        if self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = x[:,np.newaxis,...]
            x = torch.from_numpy(x).float() 
            return x

        elif self.cache == 'in_memory':
            return x

@register('mercator')
class Mercator(Dataset):

    def __init__(self, root_path, init_year, last_year, first_k=None,
                 repeat=1, cache='bin'):
        self.repeat = repeat
        self.cache = cache

        self.files = []
        ds_list = []

        for iyear,year in enumerate(np.arange(init_year,last_year+1)) :
            file_regexpr = f'glorys12v1_mod_product_001_030_{year}-*.nc'
            mult_datafile = os.path.join(root_path,file_regexpr)
            tmp_ds = xr.open_mfdataset(mult_datafile)
            #
            # quelque traitement: selection (slice), changement de resolution, retrait des NaN, ...
            #
            ds_list.append(tmp_ds)

        #Loading MDT (Mean Dynamic Topology)
        datapath = '/data/jean.legoff/data/Copernicus/OutputData/data_LON-64-42-LAT+26+44/'
        extrapath = 'GLORYS12V1_PRODUCT_001_030-extra'
        extra_datafile = os.path.join(datapath,extrapath,'glorys12v1_mdt_mod_product_001_030.nc')
        mdt_ds = xr.open_mfdataset(extra_datafile)

        # Concatenation des Datasets en un seul
        #print(ds_list)    
        #ds = xr.concat(ds_list, dim='time', join='override')
        ds = xr.concat(ds_list, dim='time')
        n_time, n_lat, n_lon = ds.dims['time'], ds.dims['latitude'], ds.dims['longitude']
        npz_data = ds['sla'][:,:min(n_lat,n_lon),:min(n_lat,n_lon)].to_numpy()+mdt_ds['mdt'][:min(n_lat,n_lon),:min(n_lat,n_lon)].to_numpy()

        for t in range(n_time):

            #if cache == 'none':
             #   self.files.append(file)

            if cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, str(t) + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(npz_data[t], f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    npz_data[t]))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        #if self.cache == 'none':
         #   return transforms.ToTensor()(Image.open(x).convert('RGB'))

        if self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = x[np.newaxis,...]
            x = torch.from_numpy(x).float() 
            return x

        elif self.cache == 'in_memory':
            return x

