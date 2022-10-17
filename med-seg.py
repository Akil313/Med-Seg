# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 18:59:55 2022

@author: itchi
"""

import random
from unet_model import build_unet
from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from patchify import patchify, unpatchify

image_stack= tiff.imread("images/training.tif")
mask_stack= tiff.imread("images/training_groundtruth.tif")

all_image_patches = []
for img in range(image_stack.shape[0]):
    
    large_image = image_stack[img]
    
    patches_img = patchify(large_image, (256, 256), step=256)
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            single_patch_img = (single_patch_img.astype('float32'))/255
            
            all_image_patches.append(single_patch_img)
            
images = np.array(all_image_patches)
images = np.expand_dims(images, -1)

all_mask_patches = []
for img in range(mask_stack.shape[0]):
    
    large_mask = mask_stack[img]
    
    patches_mask = patchify(large_mask, (256, 256), step=256)
    
    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            
            single_patch_mask = patches_mask[i,j,:,:]
            single_patch_mask = single_patch_mask /255
            
            all_mask_patches.append(single_patch_mask)
            
masks = np.array(all_mask_patches)
masks = np.expand_dims(masks, -1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.25, random_state = 0)


image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()


IMG_HEIGHT = images.shape[1]
IMG_WIDTH = images.shape[2]
IMG_CHANNELS = images.shape[3]

print(images.shape)


input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = build_unet(input_shape)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()


seed=24
from keras.preprocessing.image import ImageDataGenerator

y_pred=model.predict(X_train[0])