#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Developed by: Ivan Legorreta
Contact information: ilegorreta@outlook.com
'''

import nibabel as nib
import os
import numpy as np
import h5py

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'

dataDirNormal = "normal"
dataDirAbnormal = "abnormal"

#Step 1: Loading the volumes into a numpy array
volumes = np.zeros([5051,217,178,60,1], dtype='float32')
print(volumes.dtype)

cont = 0
print("Begining...")
for img in sorted(os.listdir(dataDirNormal)):
  example_filename = os.path.join(dataDirNormal, img)
  image = nib.load(example_filename)
  data = image.get_fdata()
  data = np.float32(data)
  data = np.nan_to_num(data)
  data[data < 0] = 0
  data[data >= 10000] = 10000
  data = np.expand_dims(data, axis=3)
  pad = np.zeros([217, 178, 60, 1], dtype='float32')
  pad[:data.shape[0],:data.shape[1],:data.shape[2]] = data
  volumes[cont] = pad
  print(cont)
  cont = cont + 1

print("Normal Badge Finished! Number of volumes: " + str(cont))

for img in sorted(os.listdir(dataDirAbnormal)):
  example_filename = os.path.join(dataDirAbnormal, img)
  image = nib.load(example_filename)
  data = image.get_fdata()
  data = np.float32(data)
  data = np.nan_to_num(data)
  data[data < 0] = 0
  data[data >= 10000] = 10000
  data = np.expand_dims(data, axis=3)
  pad = np.zeros([217, 178, 60, 1], dtype='float32')
  pad[:data.shape[0],:data.shape[1],:data.shape[2]] = data
  volumes[cont] = pad
  print(cont)
  cont = cont + 1

print("Finished! Final volumes shape: " + str(volumes.shape))

#Data normalization
min1 = np.amin(volumes)
max1 = np.amax(volumes)
print('Min: ',min1)
print('Max: ',max1)
volumes = (volumes - min1) / (max1 - min1)
min1 = np.amin(volumes)
max1 = np.amax(volumes)
print('New Min1: ',min1)
print('New Max1: ',max1)

print("Saving dataset into h5py file")
with h5py.File('cnn_Dataset_32.hdf5', 'w') as f:
  f.create_dataset("dataset_x", data=volumes, dtype='float32', compression="gzip")
