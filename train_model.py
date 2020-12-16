#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import os
import numpy as np
import pandas as pd
import h5py
import time
import csv
from tensorflow import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, MaxPooling3D, GlobalAveragePooling3D, Add
from keras.layers import Dropout, Input, BatchNormalization, Activation
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.losses import mean_squared_error, huber_loss
from keras.optimizers import Adadelta, SGD, Adam
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold, KFold
from keras import backend as K
started_at = time.asctime()
from scipy.stats import spearmanr
from resnet_model import model_architecture


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()

# Define the Huber loss so that it can be used with Keras
def huber_loss_wrapper(**huber_loss_kwargs):
    def huber_loss_wrapped_function(y_true, y_pred):
        return huber_loss(y_true, y_pred, **huber_loss_kwargs)
    return huber_loss_wrapped_function

#Step 1: Loading features to np arrays
with h5py.File('cnn_Dataset_32.hdf5', 'r') as f:
  ls = list(f.keys())
  print("List of Datasets in this file: \n", ls)
  X = f.get("dataset_x")
  X = np.asarray(X, dtype=np.float32)
  print(X.dtype)
  print("X shape: " , X.shape)

#Step 2: Loading labels to np arrays
results = []
with open("avg_scores.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader: # each row is a list
        results.append(row)
y = np.asarray(results, dtype=np.float32)

#Step 3: Loading sample weights
results2 = []
with open("sample_weights.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader: # each row is a list
        results2.append(row)
sample_w = np.asarray(results2, dtype=np.float32)
sample_weights = np.squeeze(sample_w, axis=1)

numEpochs = 300

bins = np.linspace(0, 1, 5)
y_binned = np.digitize(y, bins)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cont = 1
for train_val, test in list(kfold.split(X, y_binned)):
	
	y_binned_in = np.digitize(y[train_val], bins)
	in_fold = list(kfold.split(X[train_val], y_binned_in))[0]
	
	train = train_val[in_fold[0]]
	val = train_val[in_fold[1]]

	print(y[train].shape)
	print(y[val].shape)
	print(y[test].shape)
	
	#Step 4:Create the CNN Model
	model = model_architecture()

	#model.summary()
	print("Model created successfully!!")

	#Step 5: Compiling the model
	model.compile(loss=huber_loss_wrapper(delta=0.15), optimizer=Adam(lr = 0.0001), metrics=['mean_absolute_error'])
	print("Model compiled successfully!!")

	#Step 6: Checkpoint
	filepath = "weights_resnet_sw3_k%s.hdf5" % cont
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

	#Step 7: Early Stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

	#Step 6: Training the model
	history = model.fit(x=X[train], y=y[train], batch_size=16, epochs=numEpochs, validation_data=(X[val],y[val]), verbose=0, callbacks=[checkpoint, es], sample_weight=sample_weights[train])
	print("Model trained successfully!!")

	#Load best weights of this fold
	model.load_weights("weights_resnet_sw3_k%s.hdf5" % cont)
	
	#Evaluate fold
	score = model.evaluate(x=X[test], y=y[test], batch_size=16, verbose=1)
	print('val_huber_loss:, val_mae: ', score)

	#Making predictions on the test set
	print('Predicting dataset:')
	prediction = model.predict(X[test], verbose=1)

	# Spearman correlation 
	spearman_cor, p_spearman = spearmanr(y[test], prediction)
	print(f"Spearman Correlation Coefficient: {spearman_cor}")
	print(f"P value: {p_spearman}")

	#Step 7: Save results to CSV
	hist_df 		= pd.DataFrame(history.history) 
	hist_df['epoch']	= range(1,(hist_df.shape[0]+1))
	hist_df['num_epochs']   = numEpochs
	hist_df['val_huber_loss']     = score[0]
	hist_df['val_MAE']     = score[1]

	hist_df.to_csv('model_training_hist_restnet_sw3.csv', mode = 'a')

	cont += 1

	K.clear_session()
	tf.compat.v1.reset_default_graph()



