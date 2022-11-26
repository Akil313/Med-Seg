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
test_image_shepp = cv2.imread("images/online_image.png", 0)
test_image_shepp = cv2.resize(test_image_shepp, (256, 256), interpolation = cv2.INTER_AREA)
test_image_shepp = np.expand_dims(test_image_shepp, -1)
test_image_shepp = np.expand_dims(test_image_shepp, 0)

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


test_image_shepp = cv2.imread("images/online_image.png", 0)
test_image_shepp = cv2.resize(test_image_shepp, (256, 256), interpolation = cv2.INTER_AREA)
test_image_shepp = np.expand_dims(test_image_shepp, 0)
test_image_shepp = np.repeat(test_image_shepp, 100, axis=0)
test_image_shepp = np.expand_dims(test_image_shepp, -1)

test_image_shepp_mask = cv2.imread("images/factor_3_small_white_region_only.png", 0)
test_image_shepp_mask = cv2.resize(test_image_shepp_mask, (256, 256), interpolation = cv2.INTER_AREA)
test_image_shepp_mask = np.expand_dims(test_image_shepp_mask, 0)
test_image_shepp_mask = np.repeat(test_image_shepp_mask, 100, axis=0)
test_image_shepp_mask = np.expand_dims(test_image_shepp_mask, -1)

X_train = X_test = test_image_shepp
y_train = y_test = test_image_shepp_mask

from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(test_image_shepp, test_image_shepp_mask, test_size=0.25, random_state = 0)


image_number = random.randint(0, len(X_train)-1)
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

img_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

mask_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')
                     #preprocessing_function = lambda x: np.where(x<1, 0, 1).astype(x.dtype)) #Binarize the output again. 

image_data_generator = ImageDataGenerator(**img_data_gen_args)
#image_data_generator.fit(X_train, augment=True, seed=seed)

batch_size= 8

image_generator = image_data_generator.flow(X_train, seed=seed, batch_size=batch_size)
valid_img_generator = image_data_generator.flow(X_test, seed=seed, batch_size=batch_size) #Default batch size 32, if not specified here

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
#mask_data_generator.fit(y_train, augment=True, seed=seed)
mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed, batch_size=batch_size)  #Default batch size 32, if not specified here

def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

x = image_generator.next()
y = mask_generator.next()

test_mask_1 = y[1,:,:]

for i in range(0,5):
    image = x[i]
    mask = y[i]
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()


steps_per_epoch = 3*(len(X_train))//batch_size


history = model.fit(my_generator, validation_data=validation_datagen, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=steps_per_epoch, epochs=20)

print(history)


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#Predict on a few images
#model = get_model()
#model.load_weights('mitochondria_50_plus_100_epochs.hdf5') #Trained for 50 epochs and then additional 100
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')  #Trained for 50 epochs

test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)


pre_pred = model.predict(test_image_shepp)[0,:,:,0]
prediction = (model.predict(test_image_shepp)[0,:,:,0] > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()

test_image_shepp_mask = cv2.imread("images/factor_3_small_white_region_only.png", 0)
test_image_shepp_mask = cv2.resize(test_image_shepp_mask, (256, 256), interpolation = cv2.INTER_AREA)
test_image_shepp_mask[175:, 175:] = 0
plt.imshow(test_image_shepp_mask, cmap='gray')

print(test_image_shepp_mask[175:, 175:].shape)