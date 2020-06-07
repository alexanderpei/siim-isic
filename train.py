from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

imageTargetSize = (256, 256)
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

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
    trainIm,
    steps_per_epoch=2000 // batchSize,
    epochs=1,
    validation_data=valIm,
    validation_steps=800 // batchSize)

testNames = testIm.filenames
nTest = len(testNames)
ytest = model.predict_generator(testIm, steps=nTest)

print(ytest)
print(ytest.shape)