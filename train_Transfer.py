import tensorflow as tf
import pandas as pd
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras.applications.resnet50 import preprocess_input

import efficientnet.tfkeras as efn
# from efficientnet import preprocessing as preprocess_input

# Check if on google colab

if os.getcwd() == '/content/siim-isic':
    pathBase = '/content/drive/My Drive/KaggleData/dataset-siim-isic/'
else:
    pathBase = os.getcwd()
# Specify the input and batch size

imageTargetSize = 256, 256
batchSize = 4
train = True
tf.random.set_seed(1234)


# Data generators for the image directories

datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_input,
            fill_mode='nearest')

testgen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

trainIm = datagen.flow_from_directory(
    os.path.join(pathBase, 'data', 'train'),
    target_size=imageTargetSize,
    batch_size=batchSize,
    class_mode='binary')

valIm = datagen.flow_from_directory(
    os.path.join(pathBase, 'data', 'val'),
    target_size=imageTargetSize,
    batch_size=batchSize,
    class_mode='binary')

testIm = testgen.flow_from_directory(
    os.path.join(pathBase, '512x512-test'),
    target_size=imageTargetSize,
    batch_size=batchSize,
    shuffle=False,
    class_mode='binary')

# Get number of positive and negative classes, and then calculate bias

# ytrain = []
#
# for i in range(len(trainIm.filenames)):
#     ytrain.extend(np.array(trainIm[i][1]))
#
# pos = np.sum(ytrain == 1)
# neg = np.sum(ytrain == 0)
# bias = np.log([pos/neg])
#
# print(pos, neg)

# Set up transfer learning architecture

base_model = efn.EfficientNetB5(weights='imagenet', include_top=False, input_shape=(*imageTargetSize, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(
    lr=0.0001)

#loss = [focal_loss(alpha=0.25, gamma=2)]
loss = tf.keras.losses.BinaryCrossentropy()

class_weight = {0: 0.1, 1: 0.9} # Can try using class weights to fix bias in the data

model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['accuracy', tf.keras.metrics.AUC()])

# Callback function for saving progress

if not os.path.isdir('./checkpoint/'):
    os.mkdir('./checkpoint/')

checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 monitor='val_auc',
                                                 mode='max',
                                                 save_best_only=True,
                                                 verbose=1)

# Train

if train:
    model.fit(
        trainIm,
        class_weight=class_weight,
        steps_per_epoch=2000 // batchSize,
        epochs=8,
        validation_data=valIm,
        validation_steps=800 // batchSize,
        callbacks=[cp_callback])

# Test and create output CSV

model.load_weights(checkpoint_path)

df_test = pd.DataFrame({
    'image_name': os.listdir(os.path.join(os.getcwd(), '512x512-test', '512x512-test'))
})

df_test['image_name'] = df_test['image_name'].str.split('.').str[0]
print(df_test.shape)
df_test.head()

testNames = testIm.filenames
nTest = len(testNames)
ytest = model.predict(testIm, steps=np.ceil(float(nTest) / float(batchSize)))

df_test['target'] = ytest
df_test.to_csv('submission.csv', index=False)

model.evaluate(trainIm, steps=np.ceil(float(nTest) / float(batchSize)))