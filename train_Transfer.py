import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import os

from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
import efficientnet.tfkeras as efn
from focal_loss import focal_loss

# Check if on google colab. If not, set the base directory to the current one.
if os.getcwd() == '/content/siim-isic':
    pathBase = '/content/drive/My Drive/datasets/siim-isic/'
else:
    pathBase = os.getcwd()

# Specify the input and batch size, as well as some other parameters. The original image size is 512x512. We will
# set the image size to 256x256 for computational speed.
#
# Batch size is 4. With larger batches, computational speed will increase, however must GPUs won't be able to have
# a large batch size. Having smaller batch sizes also increase stochasticity during learning. Consider that a bug or a
# feature.
#
# train = True will either run model.fit() or not.
#
# nEpochs are the number of training epochs.
#
# lr is the learning rate. This should start between 0.001 and 0.00001.

imageTargetSize = 256, 256
batchSize = 4
train = True
nEpochs = 20
lr = 0.0005
# tf.random.set_seed(42069)

# Choose your optimizer function. For example, try tf.keras.optimizers.sgd()
# Lookahead has been shown to reduce variance during training (https://arxiv.org/abs/1907.08610) . It is optional.
opt = tf.keras.optimizers.Adam(
    lr=lr)
opt = tfa.optimizers.Lookahead(opt)

# Set up transfer learning architecture. We are using a pre-trained model to do transfer learning. Feel
# free to change the base model to whatever model you like.
baseModel = efn.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(*imageTargetSize, 3))

# Whatever loss function you wish to try. Focal loss has been shown to work well (https://arxiv.org/abs/1708.02002).
# Or you can try to use BCE.
#loss = [focal_loss(alpha=0.25, gamma=2)]
loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)

# Can try using class weights to fix bias in the data. Down-weighting the benign class since there are more of them.
class_weight = {0: 0.1, 1: 0.9}

# Need to get statistics of data to do feature wise mean and feature wise std. Will randomly sample 1000 images in the
# training data set for ImageDataGenerator().fit(). We should find the statistics for the entire 60,000 images,
# but your computer will probably run out of memory.
file_path = os.path.join(os.getcwd(), '512x512-dataset-melanoma', '512x512-dataset-melanoma')
nFiles = len(os.listdir(file_path))
nIdx = 1000
x = np.zeros((nIdx, *imageTargetSize, 3), dtype=np.int8)

idx = np.random.randint(0, nFiles, size=(nIdx))

for i in range(nIdx):
    filename = os.listdir(file_path)[idx[i]]
    img = Image.open(os.path.join(file_path, filename))
    img = img.resize((imageTargetSize))
    x[i, :, :, :] = np.asarray(img, dtype=np.int8)

# Data generators for the image directories.
# Images are randomly rotated, shifted, and flipped to increase training generalization.
#
# trainGen is the generator for train and validation data, testGen is the generator for the training data which
# does not require data augmentation.

trainGen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            brightness_range=[0.6,1.0],
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

testGen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True)

# Need to fit the mean and variance of the 1000 randomly sampled images
trainGen.fit(x)
testGen.fit(x)
del x

trainIm = trainGen.flow_from_directory(
    os.path.join(pathBase, 'data', 'train'),
    target_size=imageTargetSize,
    batch_size=batchSize,
    class_mode='binary')

valIm = trainGen.flow_from_directory(
    os.path.join(pathBase, 'data', 'val'),
    target_size=imageTargetSize,
    batch_size=batchSize,
    class_mode='binary')

testIm = testGen.flow_from_directory(
    os.path.join(pathBase, '512x512-test'),
    target_size=imageTargetSize,
    batch_size=batchSize,
    shuffle=False,
    class_mode='binary')


# Adding a few extra layers on top of the base model that we can train.

model = Sequential()
model.add(baseModel)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='selu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='selu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='selu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='selu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['accuracy', tf.keras.metrics.AUC()])

# Creating the path for the checkpoint. Keep looping until is not a path. Callback function for saving progress
cpCount = 0
pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount))

while os.path.isdir(pathCP):
    cpCount += 1
    pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount))

os.makedirs(pathCP)
checkpoint_path = os.path.join(pathCP, 'cp.ckpt')

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

# Learn rate scheduler. Decrease learning rate over time

def scheduler(epoch):
  if epoch < 6:
    return lr
  else:
    return lr * tf.math.exp(0.2 * (6 - epoch))

sc_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Train

if train:
    model.fit(
        trainIm,
        class_weight=class_weight,
        steps_per_epoch=2000 // batchSize,
        epochs=nEpochs,
        verbose=1,
        validation_data=valIm,
        validation_steps=800 // batchSize,
        callbacks=[cp_callback, sc_callback, csv_callback])

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
yTest = model.predict(testIm, steps=np.ceil(float(nTest) / float(batchSize)))

df_test['target'] = yTest
nameOut = 'submission' + str(cpCount) + '.csv'

# For some reason, things get scrambled in the cluster supercomputer. Need to sort.
df_test['image_name'] = df_test['image_name'].sort_values(ascending=True).values
df_test.to_csv(os.path.join(pathCP, nameOut), index=False)

