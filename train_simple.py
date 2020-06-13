import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D

# Check if on google colab. If not, set the base directory to the current one.
if os.getcwd() == '/content/siim-isic':
    pathBase = '/content/drive/My Drive/datasets/siim-isic/'
else:
    pathBase = os.getcwd()

# ------------------ Initialize Parameters ------------------ #

h = w = 256     # Image height and width to convert to. 256x256 is good for memory and performance.
batchSize = 4   # Batch size. Higher batch size speeds up, but will cost more memory and be less stochastic.
nEpochs = 3     # Number of training epochs.
lr = 0.0005     # Learning rate
foldNum = 0     # K-fold number you want to use (0-4)

# ------------------ Creating model ------------------ #

# Choose your optimizer function. For example, try SGD
opt = Adam(lr=lr)

# Choose your loss function. Can try BinaryCrossentropy(label_smoothing=0.1)
loss = BinaryCrossentropy()

# Adding the desired model layers
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(h, w, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile the model
model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['accuracy', tf.keras.metrics.AUC()]) # Print the accuracy and AUC during training

# ------------------ Gather Data ------------------ #

# Creating the image data generators to augment the data. This will make sure that the model isn't just
# memorizing the images. We create a new test generator since we don't want to change around the test images.

trainGen = ImageDataGenerator(
            rescale=1./225,         # Scale down the image which should improve learning gradients
            rotation_range=180,     # Randomly rotate the image from -180 to 180 degrees
            width_shift_range=0.2,  # Randomly shift the width of the image
            height_shift_range=0.2, # "
            horizontal_flip=True,   # Randomly flip the image horizontally
            vertical_flip=True,     # "
            fill_mode='nearest')    # Fill Nan values with the nearest pixel

testGen = ImageDataGenerator(
            rescale=1./225,
            fill_mode='nearest')

trainIm = trainGen.flow_from_directory(
            os.path.join(pathBase, 'data' + str(foldNum), 'train'),
            target_size=(h, w),
            batch_size=batchSize,
            class_mode='binary')

valIm = trainGen.flow_from_directory(
            os.path.join(pathBase, 'data' + str(foldNum), 'val'),
            target_size=(h, w),
            batch_size=batchSize,
            class_mode='binary')

testIm = testGen.flow_from_directory(
            os.path.join(pathBase, '512x512-test'),
            target_size=(h, w),
            batch_size=batchSize,
            shuffle=False,
            class_mode='binary')

# ------------------ Train the model ------------------ #

# Need to specify how many batches we want to use during training
steps_per_epoch = np.ceil(float(len(trainIm.filenames)) / float(batchSize))
validation_steps = np.ceil(float(len(valIm.filenames)) / float(batchSize))

model.fit(
    trainIm,
    steps_per_epoch=steps_per_epoch,
    epochs=nEpochs,
    verbose=1,
    validation_data=valIm,
    validation_steps=validation_steps)

# ------------------ Do Predictions ------------------ #

print('Generating predictions and submission file...')

df_test = pd.DataFrame({
    'image_name': os.listdir(os.path.join(os.getcwd(), '512x512-test', '512x512-test'))
})
df_test['image_name'] = df_test['image_name'].str.split('.').str[0] # Removes .jpg extension

testNames = testIm.filenames
nTest = len(testNames)

yTest = model.predict(testIm, steps=np.ceil(float(nTest) / float(batchSize)))

df_test['target'] = yTest
nameOut = 'submission.csv'
df_test.to_csv(os.path.join(pathBase, nameOut), index=False)

