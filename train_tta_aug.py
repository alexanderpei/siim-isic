import os
import sys
import numpy as np
import pandas as pd
import albumentations as A
import tensorflow as tf
import tensorflow_addons as tfa
import efficientnet.tfkeras as efn

from PIL import Image
from pp_MoveIm import moveim
from focal_loss import focal_loss
from ImageDataAugmentor.image_data_augmentor import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D

# Check if on google colab. If not, set the base directory to the current one.
if os.getcwd() == '/content/siim-isic':
    pathBase = '/content/drive/My Drive/datasets/siim-isic/'
else:
    pathBase = os.getcwd()

# If you didn't move the images
if not os.path.isdir('data_split'):
    moveim()

# ------------------ Initialize parameters ------------------ #

h = w = 256     # Image height and width to convert to. 256x256 is good for memory and performance.
batchSize = 4  # Batch size. Higher batch size speeds up, but will cost more memory and be less stochastic.
nEpochs = 4   # Number of training epochs.
lr = 0.0001     # Learning rate
base = 'b1'     # Base model you want to use. Should be some type of EfficientNet or ResNet.
ntta = 2       # Number of time-test augmentations to do.
ntop = 2        # Number of top epoch model checkpoints to load.

# foldNum can be specified via the command line. This will indicate which k-fold split you want to use for training.
try:
    foldNum = sys.argv[1]
except (NameError, IndexError):
    foldNum = 0

print('Using k-fold: ' + str(foldNum))

# ------------------ Creating model ------------------ #

# Choose your optimizer function. For example, try tf.keras.optimizers.sgd()
# Lookahead has been shown to reduce variance during training (https://arxiv.org/abs/1907.08610) . It is optional.
opt = tf.keras.optimizers.Adam(
    lr=lr)
opt = tfa.optimizers.Lookahead(opt)

# loss = [focal_loss(alpha=0.25, gamma=2)]
loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)

# Can try using class weights to fix bias in the data. Down-weighting the benign class since there are more of them.
class_weight = {0: 0.2, 1: 0.8}


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

cpCount = 0
pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount) + '_' + str(foldNum))

while os.path.isdir(pathCP):
    cpCount += 1
    pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount) + '_' + str(foldNum))

os.makedirs(pathCP)

# Create a callback that saves the model's weights
checkpoint_path = os.path.join(pathCP, 'cp-{epoch:04d}.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create a callback that logs the training progress
csvOut = os.path.join(pathCP, 'training.log')
csv_callback = CSVLogger(csvOut)

# Learn rate scheduler. Decrease learning rate over time, or use reduce LR on plateau. Monitor by the val_auc.

# def scheduler(epoch):
#   if epoch < 6:
#     return lr
#   else:
#     return lr * tf.math.exp(0.2 * (6 - epoch))
#
# sc_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

sc_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.6,
    patience=4,
    min_lr=5e-6)

# ------------------ Gather and augment data ------------------ #

# Augmentations using Albumentations https://github.com/albumentations-team/albumentations
aug = A.Compose([
    A.Flip(p=1),
    A.RandomGamma(gamma_limit=(100, 200), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(rotate_limit=180, scale_limit=(0, 1), p=0.5),
    A.Rotate(limit=180, p=1),
    # A.RGBShift(p=0.5),
    A.GaussNoise(p=0.5),
    # A.CenterCrop(height=100, width=100, p=1),
    A.CoarseDropout(max_holes=10, min_holes=5, max_width=4, max_height=200, fill_value=0, p=0.5),
    A.CoarseDropout(max_holes=10, min_holes=5, max_width=200, max_height=4, fill_value=0, p=0.5),
    A.Normalize(p=1.0),
    # A.HueSaturationValue(p=0.5),
    # A.ChannelShuffle(p=0.5),
    A.MedianBlur(p=0.5, blur_limit=5),
])

# Using the same generator since we are also augmenting the test data images for test-time augmentation.
# This is a special generator which can receive Albumentations augmentations
# https://github.com/mjkvaak/ImageDataAugmentor
imGen = ImageDataAugmentor(
    augment=aug)

trainIm = imGen.flow_from_directory(
    os.path.join(pathBase, 'data_split', 'data' + str(foldNum), 'train'),
    target_size=(h, w),
    batch_size=batchSize,
    class_mode='binary')

valIm = imGen.flow_from_directory(
    os.path.join(pathBase, 'data_split', 'data' + str(foldNum), 'val'),
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
steps_per_epoch = np.ceil(float(len(trainIm.filenames)) / float(batchSize) / 200)
validation_steps = np.ceil(float(len(valIm.filenames)) / float(batchSize) / 200)

model.fit(
    trainIm,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weight,
    epochs=nEpochs,
    verbose=2,
    validation_data=valIm,
    validation_steps=validation_steps,
    callbacks=[cp_callback, sc_callback, csv_callback])

# ------------------ Do TTA predictions and forward pass ------------------ #

# After training the model, we will pass all of the training and test images through the model and generate a
# prediction. This prediction will be merged with other models for an ensembled prediction.

# Saving to a folder called 'forward'
pathOut = os.path.join(os.getcwd(), 'forward', 'cp_' + str(cpCount) + '_' + str(foldNum))
if not os.path.isdir(pathOut):
    os.makedirs(pathOut)

# Retrieving the n highest training checkpoints
df_log = pd.read_csv(os.path.join(pathCP, 'training.log'))
df_log = df_log.nlargest(n=ntop, columns='val_auc')
idx = df_log['epoch'].to_numpy() + 1

# Empty df with the image names in the train and test set
df_test = pd.DataFrame({
    'image_name': os.listdir(os.path.join(os.getcwd(), '512x512-test', '512x512-test'))
})
df_test['image_name'] = df_test['image_name'].str.split('.').str[0]  # Removes .jpg extension

df_train = pd.DataFrame({
    'image_name': os.listdir(os.path.join(os.getcwd(), '512x512-dataset-melanoma', '512x512-dataset-melanoma'))
})
df_train['image_name'] = df_train['image_name'].str.split('.').str[0]  # Removes .jpg extension

# Number of train and test file names
nTest = df_test.shape[0]
nTrain = df_train.shape[0]

# Now we will loop through every epoch. Each epoch will have n number of test time augmentations.
c = 0
for epoch in idx:
    # Formatting to load the checkpoint file
    stre = str(epoch)
    ncp = stre.zfill(4)
    checkpoint_path = os.path.join(pathCP, 'cp-' + ncp + '.ckpt')

    # Load the weights of this checkpoint into the model
    model.load_weights(checkpoint_path)

    yTest = np.zeros((nTest, 1))
    yTrain = np.zeros((nTrain, 1))

    # Now we will augment the data for ntta amount of times to change around the images
    for i in range(ntta):

        trainIm = imGen.flow_from_directory(
            os.path.join(pathBase, '512x512-dataset-melanoma'),
            target_size=(h, w),
            batch_size=batchSize,
            shuffle=False,
            class_mode='binary')

        testIm = imGen.flow_from_directory(
            os.path.join(pathBase, '512x512-test'),
            target_size=(h, w),
            batch_size=batchSize,
            shuffle=False,
            class_mode='binary')

        yTest += model.predict(testIm, steps=np.ceil(float(nTest) / float(batchSize))) / ntta
        yTrain += model.predict(trainIm, steps=np.ceil(float(nTrain) / float(batchSize))) / ntta

    df_test['target'] = yTest
    df_train['target'] = yTrain

    # Need to sort, sometimes the computer cluster will mess things up
    df_test['image_name'] = df_test['image_name'].sort_values(ascending=True).values
    df_train['image_name'] = df_train['image_name'].sort_values(ascending=True).values

    nameTrain = 'train_' + str(foldNum) + '_' + str(c) + '.csv'
    nameTest = 'test_' + str(foldNum) + '_' + str(c) + '.csv'

    df_test.to_csv(os.path.join(pathOut, nameTest), index=False)
    df_train.to_csv(os.path.join(pathOut, nameTrain), index=False)
    c += 1