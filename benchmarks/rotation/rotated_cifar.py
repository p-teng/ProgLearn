#%%
import random
import tensorflow as tf
import keras
from keras import layers
from itertools import product
import pandas as pd

import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil

from joblib import Parallel, delayed
from multiprocessing import Pool

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import NeuralClassificationTransformer, TreeClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

import tensorflow as tf
from skimage.transform import rotate
from scipy import ndimage
from skimage.util import img_as_ubyte

#%%

def cross_val_data(data_x, data_y, total_cls=10):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]


    for i in range(total_cls):
        indx = idx[i]#np.roll(idx[i],(cv-1)*100)
        random.shuffle(indx)

        if i==0:
            train_x1 = x[indx[0:250],:]
            train_x2 = x[indx[250:500],:]
            train_y1 = y[indx[0:250]]
            train_y2 = y[indx[250:500]]

            test_x = x[indx[500:600],:]
            test_y = y[indx[500:600]]
        else:
            train_x1 = np.concatenate((train_x1, x[indx[0:250],:]), axis=0)
            train_x2 = np.concatenate((train_x2, x[indx[250:500],:]), axis=0)
            train_y1 = np.concatenate((train_y1, y[indx[0:250]]), axis=0)
            train_y2 = np.concatenate((train_y2, y[indx[250:500]]), axis=0)

            test_x = np.concatenate((test_x, x[indx[500:600],:]), axis=0)
            test_y = np.concatenate((test_y, y[indx[500:600]]), axis=0)


    return train_x1, train_y1, train_x2, train_y2, test_x, test_y

def LF_experiment(data_x, data_y, angle, model, granularity, reps=1, ntrees=10, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)

    if model == "dnn":
        default_transformer_class = NeuralClassificationTransformer

        network = keras.Sequential()
        network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_x)[1:]))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Conv2D(filters=254, kernel_size=(3, 3), strides = 2, padding = "same", activation='relu'))

        network.add(layers.Flatten())
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(2000, activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(2000, activation='relu'))
        network.add(layers.BatchNormalization())
        network.add(layers.Dense(units=10, activation = 'softmax'))

        default_transformer_kwargs = {
            "network": network,
            "euclidean_layer_idx": -2,
            "loss": "categorical_crossentropy",
            "optimizer": Adam(3e-4),
            "fit_kwargs": {
                "epochs": 100,
                "callbacks": [EarlyStopping(patience=5, monitor="val_loss")],
                "verbose": False,
                "validation_split": 0.33,
                "batch_size": 32,
            },
        }
        default_voter_class = KNNClassificationVoter
        default_voter_kwargs = {"k" : int(np.log2(2500))}

        default_decider_class = SimpleArgmaxAverage
    elif model == "uf":
        default_transformer_class = TreeClassificationTransformer
        default_transformer_kwargs = {"kwargs" : {"max_depth" : 30, "max_features" : "auto"}}

        default_voter_class = TreeClassificationVoter
        default_voter_kwargs = {}

        default_decider_class = SimpleArgmaxAverage


    errors = np.zeros(2)

    for rep in range(reps):
        print("Starting Rep {} of Angle {}".format(rep, angle))
        train_x1, train_y1, train_x2, train_y2, test_x, test_y = cross_val_data(data_x, data_y, total_cls=10)


        #change data angle for second task
        tmp_data = train_x2.copy()
        _tmp_ = np.zeros((32,32,3), dtype=int)
        total_data = tmp_data.shape[0]

        for i in range(total_data):
            tmp_ = image_aug(tmp_data[i],angle)
            tmp_data[i] = tmp_

        if model == "uf":
            train_x1 = train_x1.reshape((train_x1.shape[0], train_x1.shape[1] * train_x1.shape[2] * train_x1.shape[3]))
            tmp_data = tmp_data.reshape((tmp_data.shape[0], tmp_data.shape[1] * tmp_data.shape[2] * tmp_data.shape[3]))
            test_x = test_x.reshape((test_x.shape[0], test_x.shape[1] * test_x.shape[2] * test_x.shape[3]))

        

        progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                     default_transformer_kwargs = default_transformer_kwargs,
                                     default_voter_class = default_voter_class,
                                     default_voter_kwargs = default_voter_kwargs,
                                     default_decider_class = default_decider_class)

        progressive_learner.add_task(
            X = train_x1,
            y = train_y1,
            transformer_voter_decider_split = [0.67, 0.33, 0],
            decider_kwargs = {"classes" : np.unique(train_y1)}
        )

        llf_single_task=progressive_learner.predict(test_x, task_id=0)

        progressive_learner.add_task(
            X = tmp_data,
            y = train_y2,
            transformer_voter_decider_split = [0.67, 0.33, 0],
            decider_kwargs = {"classes" : np.unique(train_y2)}
        )

        llf_task1=progressive_learner.predict(test_x, task_id=0)

        errors[1] = errors[1]+(1 - np.mean(llf_task1 == test_y))
        errors[0] = errors[0]+(1 - np.mean(llf_single_task == test_y))

    errors = errors/reps
    print("Errors For Angle {}: {}".format(angle, errors))
    with open('rotation_results/'+model+'-'+str(angle)+'.pickle', 'wb') as f:
        pickle.dump(errors, f, protocol = 2)

def _image_aug(pic, angle, centroid_x=23, centroid_y=23, win=16, scale=1.45):
    im_sz = int(np.floor(pic.shape[1]*scale))
    pic_ = np.uint8(np.zeros((im_sz,im_sz,3),dtype=int))

    pic_[:,:,0] = ndimage.zoom(pic[0,:,:],scale)

    pic_[:,:,1] = ndimage.zoom(pic[1,:,:],scale)
    pic_[:,:,2] = ndimage.zoom(pic[2,:,:],scale)

    image_aug = rotate(pic_, angle, resize=False)
    #print(image_aug.shape)
    image_aug_ = image_aug[centroid_x-win:centroid_x+win,centroid_y-win:centroid_y+win,:]
    image_aug_ = image_aug_.reshape(3,32,32)

    return img_as_ubyte(image_aug_)

### MAIN HYPERPARAMS ###
model = "dnn"
granularity = 4
reps = 10
angles = range(0,184,granularity)
########################
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]

if model == "dnn":
    for angle in angles:
        LF_experiment(data_x, data_y, angle, model, granularity, reps=reps, ntrees=0, acorn=1)
elif model == "uf":
    Parallel(n_jobs=-1)(delayed(LF_experiment)(data_x, data_y, angle, model, granularity, reps=reps, ntrees=10, acorn=1) for angle in angles)
