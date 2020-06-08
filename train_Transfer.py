import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

imageTargetSize = 256, 256
batchSize = 16

datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')

testgen = ImageDataGenerator(
            rescale=1./255)

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
    batch_size=1,
    shuffle=False,
    class_mode='binary')

# Set up transfer learning architecture

new_input = Input(shape=(*imageTargetSize, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=new_input)

x = base_model.output
x = GlobalAveragePooling2D()(x)
pred = Dense(1, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=pred)





model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

model.fit_generator(
    trainIm,
    steps_per_epoch=2000 // batchSize,
    epochs=1,
    validation_data=valIm,
    validation_steps=800 // batchSize)