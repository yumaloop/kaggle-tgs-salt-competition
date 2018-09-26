import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import random

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook 
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras import backend as K
from keras import optimizers
from datetime import datetime

from utils.preprocessing import upsample, downsample, cov_to_class
from utils.figplot import plot_history
from utils.predict import predict_result
from utils.metrics import get_iou_vector, my_iou_metric, my_iou_metric_2
from models import Unet_resnet 

import time
t_start = time.time()



# PATH & DIR

img_size_ori = 101
img_size_target = 101
epochs = 2
batch_size = 32

base_dir = "/home/araya/kaggle-tgs-salt"
base_name ='Unet_resnet_v5' + datetime.now().strftime("__%Y-%m-%d-%H-%M-%S")
submission_file_name = base_name + '.csv'
log_dir =os.path.join(base_dir, 'main', 'train_log', base_name)

model_csv_path = os.path.join(log_dir, 'my_iou_metric.csv')
model_ckp_path = os.path.join(log_dir, 'model_checkpoint.h5')
model_arch_path = os.path.join(log_dir, 'model_arch.png')

if not os.path.exists(log_dir):
    os.makedirs(log_dir) 


# DATASET 

# Loading of training/testing ids and depths
train_df = pd.read_csv("./dataset/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("./dataset/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = [np.array(load_img("./dataset/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("./dataset/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

# split data
_, _, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, 
    stratify=train_df.coverage_class, 
    random_state=0)

# Data augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(y) for y in y_train], axis=0)



# model comple
model = Unet_resnet(img_width=101, img_height=101, img_ch=1, first_filters=16, dropout_rate=0.5).build_model()
model.compile(loss="binary_crossentropy", 
              optimizer=optimizers.adam(lr=0.01),
              metrics=[my_iou_metric])


model.summary()





# callbacks
csv_log = CSVLogger(model_csv_path)

model_checkpoint = ModelCheckpoint(
                                   model_ckp_path,
                                   monitor='my_iou_metric', 
                                   mode = 'max', 
                                   save_best_only=True, verbose=1)

reduce_lr = ReduceLROnPlateau(
                                   monitor='my_iou_metric', 
                                   mode = 'max',
                                   factor=0.5, 
                                   patience=5, 
                                   min_lr=0.0001, 
                                   verbose=1)


callbacks=[]
callbacks.append(model_checkpoint)
callbacks.append(reduce_lr)
callbacks.append(csv_log)




# model learning
history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1)





#from keras.utils import plot_model
#plot_model(model, to_file=model_arch_path, show_shapes=True)







# save model "INSTANCE"
f1_name = 'model_instance.h5'
f1_path = os.path.join(log_dir, f1_name)
model.save(f1_path)

# save model "WEIGHTs"
f2_name = 'model_weights.h5'
f2_path = os.path.join(log_dir, f2_name) 
model.save_weights(f2_path)











        





























