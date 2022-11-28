import random
from unet_model import build_unet
from tensorflow.keras.utils import normalize
from tensorflow.keras.callbacks import ModelCheckpoint
import os, glob, sys
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from patchify import patchify, unpatchify

test_image_shepp = []
#Read Shepp Loagn image, resize to 256x256, repeat images to perform augmentations and adjust array to fit format [# examples, width, height, 1]
for filename in sorted(glob.glob('images/s_*.png')):
  im = cv2.imread(filename, 0)
  im = cv2.resize(im, (256, 256), interpolation = cv2.INTER_AREA)
  test_image_shepp.append(im)

test_image_shepp = np.expand_dims(test_image_shepp, -1)

test_image_shepp_mask = []
#Read Shepp Loagn target, resize to 256x256, repeat images to perform augmentations and adjust array to fit format [# examples, width, height, 1]
for filename in sorted(glob.glob('images/f_*.png')):
  im = cv2.imread(filename, 0)
  im = cv2.resize(im, (256, 256), interpolation = cv2.INTER_AREA)
  test_image_shepp_mask.append(im)

test_image_shepp_mask = np.expand_dims(test_image_shepp_mask, -1)

X_train = X_test = test_image_shepp
y_train = y_test = test_image_shepp_mask

from sklearn.model_selection import train_test_split

#Display the Shep Logan image and the target image
image_number = random.randint(0, len(X_train)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.savefig('plots/in_out_img.png')

IMG_HEIGHT = test_image_shepp.shape[1]
IMG_WIDTH = test_image_shepp.shape[2]
IMG_CHANNELS = test_image_shepp.shape[3]

#Build model from unet_model structure
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = build_unet(input_shape)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

seed=24
from keras.preprocessing.image import ImageDataGenerator

#Define dictionary listing augmentations to perform on input data
img_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

#Define dictionary listing augmentations to perform on target data
mask_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     preprocessing_function = lambda x: np.where(x>1, 1, 0).astype(x.dtype)) #Binarize the output again.

#Apply image generator to input and target/mask samples
image_data_generator = ImageDataGenerator(**img_data_gen_args)

batch_size= 8

image_generator = image_data_generator.flow(X_train, seed=seed, batch_size=batch_size)
valid_img_generator = image_data_generator.flow(X_test, seed=seed, batch_size=batch_size) #Default batch size 32, if not specified here

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed, batch_size=batch_size)  #Default batch size 32, if not specified here

def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

#Display augmented input and mask samples
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
    plt.title('Augmented Image and Augmented Mask')
    plt.savefig('plots/aug_pair_{}.png'.format(i))

steps_per_epoch = 3*(len(X_train))//batch_size

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(my_generator, validation_data=validation_datagen, 
                      steps_per_epoch=steps_per_epoch, 
                      validation_steps=steps_per_epoch, epochs=20,
                      callbacks=[cp_callback])

model.save_weights('models/medseg_weights.h5')

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
plt.savefig('plots/train_val_loss.png')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/train_val_accuracy.png')

#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)


prediction = (model.predict(test_img_input) > 0.2).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction[0, :, :, 0], cmap='gray')
plt.savefig('plots/output_comparrison.png')