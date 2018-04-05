#import setGPU

import math
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Flatten, Reshape, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D, InputLayer, Conv2D, UpSampling2D, AveragePooling2D

from keras.layers.advanced_activations import PReLU

from scipy import ndimage, misc

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import class_weight

import readROOT
import copy

rng = np.random.RandomState(0)


matplotlib.rcParams["figure.figsize"] = (8.0, 5.0)
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["axes.spines.left"] = True
matplotlib.rcParams["axes.spines.bottom"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["legend.fontsize"] = 14
matplotlib.rcParams["axes.titlesize"] = 14

color_palette={"Indigo":{
                50:"#E8EAF6"},
               "Teal":{
                50:"#E0F2F1"}
              }

data_directory="../data"
labels_directory="../data"

target_digi = ["DQMData","Run 1","MuonGEMDigisV","Run summary","GEMDigisTask"]
target_rechit = ["DQMData","Run 1","MuonGEMRecHitsV","Run summary","GEMRecHitsTask"]
target = copy.copy(target_rechit)
target.append("rh_dcEta_r-1_st1")

drift_tubes_layers = pd.DataFrame()
file_list=['DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO000300.root']
for file_name in file_list:
    drift_tubes_layers=drift_tubes_layers.append(readROOT.readROOT(file_name,target),
                                                 ignore_index=True);

SMOOTH_FILTER_SIZE = 3

def median_polling(layer):
    smooth_layer = []
    for index in range(len(layer) - (SMOOTH_FILTER_SIZE-1)):
        median = np.median(layer[ index : index + SMOOTH_FILTER_SIZE ])
        smooth_layer.append(median)
    return np.array(smooth_layer)

drift_tubes_layers["content_smoothed"] = drift_tubes_layers["content"].apply(median_polling)

print("Minimum raw length: % s" % min(drift_tubes_layers["content"].apply(len)))
print("Maximum raw length: % s" % max(drift_tubes_layers["content"].apply(len)))
print("Minimum smoothed length: % s" % min(drift_tubes_layers["content_smoothed"].apply(len)))
print("Maximum smoothed length: % s" % max(drift_tubes_layers["content_smoothed"].apply(len)))

def resize_occupancy(layer):
    return misc.imresize(np.array(layer).reshape(1,-1),(1,SAMPLE_SIZE),interp="bilinear",mode="F").reshape(-1)

SAMPLE_SIZE = min(drift_tubes_layers["content"].apply(len))
drift_tubes_layers["content_resized"]=drift_tubes_layers["content"].apply(resize_occupancy)

SAMPLE_SIZE = min(drift_tubes_layers["content_smoothed"].apply(len))
drift_tubes_layers["content_smoothed_resized"]=drift_tubes_layers["content_smoothed"].apply(resize_occupancy)

def scale_occupancy(layer):
    layer = layer.reshape(-1,1)
    scaler = MaxAbsScaler().fit(layer)
    return scaler.transform(layer).reshape(1,-1)

drift_tubes_layers["content_scaled"] = drift_tubes_layers["content_resized"].apply(scale_occupancy)
drift_tubes_layers["content_smoothed_scaled"] = drift_tubes_layers["content_smoothed_resized"].apply(scale_occupancy)

def artificial_neural_network():
    model = Sequential()
    model.add(Reshape((47,1), input_shape=(47,), name="input_ann"))
    model.add(Flatten(name="flatten_ann"))
    model.add(Dense(8, name="dense_ann", activation="relu"))
    model.add(Dense(2, activation="softmax", name="output_ann"))
    return model

def convolutional_neural_network():
    model = Sequential()
    model.add(Reshape((47,1), input_shape=(47,), name="input_cnn"))
    model.add(Conv1D(10,3,strides=1,padding="valid",name="convolution_cnn",activation="relu"))
    model.add(MaxPooling1D(pool_size=5,strides=5,padding="valid",name="polling_cnn"))
    model.add(Dense(8, name="dense_cnn",activation="relu"))
    model.add(Dense(2, activation="softmax", name="output_cnn"))
    return model

ann = artificial_neural_network()
cnn = convolutional_neural_network()
ann.summary()
cnn.summary()

bottleneck = 100

# Simple autoencoder

_input = Input(shape=(12,46,1), name="Input_Image")

x=Flatten(name="Flatten")(_input)
x=Dense(bottleneck, name="encoded")(x)
x=PReLU()(x)
x=Dense(12*46, name="Flatten2", activation="sigmoid")(x)
decoded=Reshape((12,46,-1),name="Reshape")(x)

autoencoder_simple=Model(_input,decoded)

#Convolutional Neural Network CNN

_input = Input(shape=(12,46,1),name="Input_Image")

x = Conv2D(4, (4,4), padding="same", name="Convolution_1")(_input)
x = PReLU(name="Activation_1")(x)
x = AveragePooling2D((4,4), padding="same", name="Polling_1")(x)
x = Flatten(name="Flatten")(x)
x = Dense(12, name="Dense1")(x)
x = PReLU(name="encoded")(x)
x = Dense(144, name="Flatten2")(x)
x = PReLU()(x)
x = Reshape((3,12,-1), name="Reshape")(x)
x = UpSampling2D((4,4))(x)
x = Conv2D(4, (4,4), padding="same")(x)
x = PReLU()(x)
decoded = Conv2D(1, (1,3), activation="sigmoid", padding="valid")(x)

autoencoder_convolution = Model(_input, decoded)
