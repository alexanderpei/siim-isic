import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
import efficientnet.tfkeras as efn

# Check if on google colab

if os.getcwd() == '/content/siim-isic':
    pathBase = '/content/drive/My Drive/datasets/siim-isic/'
else:
    pathBase = os.getcwd()

# Specify the input and batch size

imageTargetSize = 256, 256
batchSize = 8
train = True
cpCount = 0
tf.random.set_seed(42069)

# Creating the path for the checkpoint. Keep looping until is not a path. Callback function for saving progress
pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount))
checkpoint_path = os.path.join(pathCP, 'cp.ckpt')

# Data generators for the image directories. Using Resnet preprocess function "preprocess_input" for the images.
# Images are randomly rotated, shifted, and flipped to increase training generalization.
#
# trainGen is the generator for train and validation data, testGen is the generator for the training data which
# does not require data augmentation.

testGen = ImageDataGenerator(
            preprocessing_function=preprocess_input)

testIm = testGen.flow_from_directory(
    os.path.join(pathBase, '512x512-test'),
    target_size=imageTargetSize,
    batch_size=batchSize,
    shuffle=False,
    class_mode='binary')

# Set up transfer learning architecture. We are using a pre-trained model to do transfer learning. Feel
# free to change the base model to whatever model you like.

baseModel = efn.EfficientNetB5(weights='imagenet', include_top=False, input_shape=(*imageTargetSize, 3))

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

# Whatever optimizer you want to try, as well as the learning rate.
opt = tf.keras.optimizers.Adam(
    lr=lr)
# opt = tfa.optimizers.Lookahead(opt)

# Whatever loss function you wish to try.
#loss = [focal_loss(alpha=0.25, gamma=2)]
loss = tf.keras.losses.BinaryCrossentropy()

# Can try using class weights to fix bias in the data. Down-weighting the benign class since there are more of them.
class_weight = {0: 0.1, 1: 0.9}

# Compile the model
model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['accuracy', tf.keras.metrics.AUC()])

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
df_test.to_csv(os.path.join(pathCP, nameOut), index=False)