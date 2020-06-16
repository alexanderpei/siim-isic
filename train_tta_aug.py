import os
import sys
import numpy as np
import pandas as pd
import albumentations as A
import tensorflow as tf
import tensorflow_addons as tfa
import efficientnet.tfkeras as efn

from PIL import Image
from focal_loss import focal_loss
from ImageDataAugmentor.image_data_augmentor import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from my_aug import preprocess_input

# Check if on google colab. If not, set the base directory to the current one.
if os.getcwd() == '/content/siim-isic':
    pathBase = '/content/drive/My Drive/datasets/siim-isic/'
else:
    pathBase = os.getcwd()

# ------------------ Initialize Parameters ------------------ #

h = w = 256  # Image height and width to convert to. 256x256 is good for memory and performance.
batchSize = 4  # Batch size. Higher batch size speeds up, but will cost more memory and be less stochastic.
nEpochs = 30  # Number of training epochs.
lr = 0.0001  # Learning rate
base = 'b1'  # Base model you want to use. Should be some type of EfficientNet or ResNet.
ntta = 50  # Number of time-test augmentations to do.
train = True

# foldNum can be specified via the command line. This will indicate which k-fold split you want to use for training.
try:
    foldNum = sys.argv[1]
except (NameError, IndexError):
    foldNum = 0

print('Using k-fold: ' + str(foldNum))

cpCount = 0
pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount) + '_' + str(foldNum))

# ------------------ Creating model ------------------ #

# Choose your optimizer function. For example, try tf.keras.optimizers.sgd()
# Lookahead has been shown to reduce variance during training (https://arxiv.org/abs/1907.08610) . It is optional.
opt = tf.keras.optimizers.Adam(
    lr=lr)
opt = tfa.optimizers.Lookahead(opt)

loss = [focal_loss(alpha=0.75, gamma=2)]

def makeModel(opt, loss, base, h, w):
    if base == 'b1':
        baseModel = efn.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(h, w, 3))
    elif base == 'b3':
        baseModel = efn.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(h, w, 3))
    elif base == 'b5':
        baseModel = efn.EfficientNetB5(weights='imagenet', include_top=False, input_shape=(h, w, 3))

    model = Sequential()
    model.add(baseModel)
    model.add(GlobalAveragePooling2D())
    # model.add(Dense(1024, activation='selu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(512, activation='selu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(256, activation='selu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(128, activation='selu'))
    # model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.AUC()])

    return model


# ------------------ Adding model callbacks ------------------ #

# Creating the path for the checkpoint. Keep looping until is not a path. Callback function for saving progress

if train:
    cpCount = 0
    pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount) + '_' + str(foldNum))

    while os.path.isdir(pathCP):
        cpCount += 1
        pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount) + '_' + str(foldNum))

    os.makedirs(pathCP)

checkpoint_path = os.path.join(pathCP, 'cp-{epoch:04d}.ckpt')

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 monitor='val_auc',
                                                 mode='max',
                                                 save_best_only=True,
                                                 verbose=1)

# Log the training

csvOut = os.path.join(pathCP, 'training.log')
csv_callback = CSVLogger(csvOut)

# Learn rate scheduler. Decrease learning rate over time, or use reduce LR on plateau

# def scheduler(epoch):
#   if epoch < 6:
#     return lr
#   else:
#     return lr * tf.math.exp(0.2 * (6 - epoch))
#
# sc_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

sc_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.5,
    patience=4,
    min_lr=1e-6)

# ------------------ Gather Data ------------------ #

# Augmentations

aug = A.Compose([
    A.Flip(p=1),
    A.RandomGamma(gamma_limit=(100, 200), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(rotate_limit=180, scale_limit=0.9, p=0.5),
    A.Rotate(limit=180, p=1),
    A.RGBShift(p=0.5),
    A.GaussNoise(p=0.5),
    # A.CenterCrop(height=100, width=100, p=1),
    # A.Normalize(p=1.0),
    A.HueSaturationValue(p=0.5),
    # A.ChannelShuffle(p=0.5),
    A.MedianBlur(p=0.5, blur_limit=5),
])

# Using the same generator since we are also augmenting the test data images for test-time augmentation

imGen = ImageDataAugmentor(
    rescale=1./255,
    augment=aug)

trainIm = imGen.flow_from_directory(
    os.path.join(pathBase, 'data' + str(foldNum), 'train'),
    target_size=(h, w),
    batch_size=batchSize,
    class_mode='binary')

valIm = imGen.flow_from_directory(
    os.path.join(pathBase, 'data' + str(foldNum), 'val'),
    target_size=(h, w),
    batch_size=batchSize,
    class_mode='binary')

testIm = imGen.flow_from_directory(
    os.path.join(pathBase, '512x512-test'),
    target_size=(h, w),
    batch_size=batchSize,
    shuffle=False,
    class_mode='binary')

model = makeModel(opt, loss, base, h, w)

# Need to specify how many batches we want to use during training
steps_per_epoch = np.ceil(float(len(trainIm.filenames)) / float(batchSize) / 20)
validation_steps = np.ceil(float(len(valIm.filenames)) / float(batchSize) / 20)

if train:
    model.fit(
        trainIm,
        steps_per_epoch=steps_per_epoch,
        epochs=nEpochs,
        verbose=1,
        validation_data=valIm,
        validation_steps=validation_steps,
        callbacks=[cp_callback, sc_callback, csv_callback])

# ------------------ Do TTA Predictions ------------------ #

# Loop over the number of test-time augmentations to do.
for i in range(ntta):
    print('Generating predictions and submission file...')

    # New testIm every loop

    testIm = imGen.flow_from_directory(
        os.path.join(pathBase, '512x512-test'),
        target_size=(h, w),
        batch_size=batchSize,
        shuffle=False,
        class_mode='binary')

    # We are going to save a bunch of different submission fields and merge them later.

    # Test and create output CSV
    model.load_weights(checkpoint_path)

    df_test = pd.DataFrame({
        'image_name': os.listdir(os.path.join(os.getcwd(), '512x512-test', '512x512-test'))
    })
    df_test['image_name'] = df_test['image_name'].str.split('.').str[0]  # Removes .jpg extention

    testNames = testIm.filenames
    nTest = len(testNames)

    yTest = model.predict(testIm, steps=np.ceil(float(nTest) / float(batchSize)))

    df_test['target'] = yTest
    nameOut = 'submission' + str(i) + '.csv'

    # For some reason, things get scrambled in the cluster supercomputer. Need to sort.
    df_test['image_name'] = df_test['image_name'].sort_values(ascending=True).values
    df_test.to_csv(os.path.join(pathCP, nameOut), index=False)
