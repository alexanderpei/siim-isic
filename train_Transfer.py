import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.inception_v3 import inception_v3
from keras.applications.inception_v3 import preprocess_input

from focal_loss import focal_loss

imageTargetSize = 256, 256
batchSize = 8

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
    'data/train',
    target_size=imageTargetSize,
    batch_size=batchSize,
    class_mode='binary')

valIm = datagen.flow_from_directory(
    'data/val',
    target_size=imageTargetSize,
    batch_size=batchSize,
    class_mode='binary')

testIm = testgen.flow_from_directory(
    '512x512-test',
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

base_model = keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(*imageTargetSize, 3))
input_tensor = Input(shape=(*imageTargetSize, 3))
x = base_model(input_tensor)
x = GlobalAveragePooling2D()(x)
pred = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_tensor, outputs=pred)

opt = keras.optimizers.Adam(
    lr=0.00001)

#loss = [focal_loss(alpha=0.25, gamma=2)]
loss = ['binary_crossentropy']

class_weight = {0: 0.05, 1: 0.95} # Can try using class weights to fix bias in the data

model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['accuracy', tf.keras.metrics.AUC()])

model.fit_generator(
    trainIm,
    class_weight=class_weight,
    steps_per_epoch=2000 // batchSize,
    epochs=20,
    validation_data=valIm,
    validation_steps=800 // batchSize)

# Test

df_test = pd.DataFrame({
    'image_name': os.listdir(os.path.join(os.getcwd(), '512x512-test', '512x512-test'))
})

df_test['image_name'] = df_test['image_name'].str.split('.').str[0]
print(df_test.shape)
df_test.head()

testNames = testIm.filenames
nTest = len(testNames)
ytest = model.predict_generator(testIm, steps=np.ceil(float(nTest) / float(batchSize)))


df_test['target'] = ytest
df_test.to_csv('submission.csv', index=False)
