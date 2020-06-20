import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from scipy.stats import rankdata
from focal_loss import focal_loss

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import normalize
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from HelperFunctions import MakePlot

# Path which contains all of the forward propagations of all of the images through each of the networks.

pathIn = os.path.join(os.getcwd(), 'forward')
foldList = os.listdir(pathIn)
nModels = len(foldList)
print(nModels)

train = True
foldNum = 0

df_fold = pd.read_csv('folds_13062020.csv')
df_fold = df_fold.rename(columns={'image_id': 'image_name'})

df = pd.read_csv(r'C:\Users\Alex\PycharmProjects\siim-isic\forward\cp_3_0\train_0_0.csv')
nTrain = df.shape[0]

df_testall = pd.read_csv(r'C:\Users\Alex\PycharmProjects\siim-isic\forward\cp_3_0\test_0_0.csv')
nTest = df_testall.shape[0]

c = 0
for path in os.listdir(pathIn):

    tempTrain = np.zeros((nTrain))
    tempTest = np.zeros((nTest))

    for file in glob.glob(os.path.join(pathIn, path, 'train*')):
        df_train = pd.read_csv(file)
        tempTrain += rankdata(df_train['target'].to_numpy()) / len(glob.glob(os.path.join(pathIn, path, 'train*'))) / len(df_train.index)
        # tempTrain += df_train['target'].to_numpy() / len(glob.glob(os.path.join(pathIn, path, 'train*')))
    for file in glob.glob(os.path.join(pathIn, path, 'test*')):
        df_test = pd.read_csv(file)
        tempTest += rankdata(df_test['target'].to_numpy()) / len(glob.glob(os.path.join(pathIn, path, 'test*'))) / len(df_test.index)
        # tempTest += df_test['target'].to_numpy() / len(glob.glob(os.path.join(pathIn, path, 'test*')))

    df_train['target'] = tempTrain
    df_test['target'] = tempTest
    c += 1
    df_fold = df_fold.merge(df_train, how='left', on='image_name', suffixes=(str(c), str(c) + 'y'))
    df_testall = df_testall.merge(df_test, how='left', on='image_name', suffixes=(str(c), str(c) + 'y'))

idxVal = df_fold.index[df_fold['fold'] == foldNum]
idxTrain = df_fold.index[df_fold['fold'] != foldNum]

nVal = len(idxVal)
nTrain = len(idxTrain)

xVal = df_fold.loc[idxVal].to_numpy()
xTrain = df_fold.loc[idxTrain].to_numpy()

yVal = df_fold['target1'].loc[idxVal].to_numpy()
yTrain = df_fold['target1'].loc[idxTrain].to_numpy()

xVal = xVal[:, -nModels:].astype(np.float32)
xTrain = xTrain[:, -nModels:].astype(np.float32)

# Test data
xTest = df_testall.to_numpy()
xTest = xTest[:, -nModels:].astype(np.float32)

print(nTrain)
print(nTest)
xVal = normalize(xVal, axis=0)
xTrain = normalize(xTrain, axis=0)
xTest = normalize(xTest, axis=0)

print(xTrain.shape)
print(xVal.shape)
print(xTest.shape)
print(yVal.shape)
print(yTrain.shape)

if train:
    # Model Params

    lr = 0.001
    nEpoch = 30
    opt = tf.keras.optimizers.Adam(
        lr=lr)
    opt = tfa.optimizers.Lookahead(opt)

    loss = [focal_loss(alpha=0.75, gamma=2)]
    # loss = [focal_loss(alpha=0.25, gamma=2)]
    loss = tf.keras.losses.BinaryCrossentropy()

    # Can try using class weights to fix bias in the data. Down-weighting the benign class since there are more of them.
    class_weight = {0: 0.2, 1: 0.8}
    # Callback

    sc_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=5,
        min_lr=1e-6)

    cpCount = 0
    pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount) + '_' + str(foldNum))

    while os.path.isdir(pathCP):
        cpCount += 1
        pathCP = os.path.join(os.getcwd(), 'checkpoint', 'checkpoint' + str(cpCount) + '_' + str(foldNum))

    # Log the training

    csvOut = os.path.join(pathCP, 'training.log')
    csv_callback = CSVLogger(csvOut)

    os.makedirs(pathCP)

    checkpoint_path = os.path.join(pathCP, 'cp-{epoch:04d}.ckpt')

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

# Train

    model = Sequential()
    model.add(Dense(20, input_dim=nModels, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.AUC()])

    history = model.fit(
                xTrain,
                yTrain,
                validation_data=(xVal, yVal),
                epochs=nEpoch,
                batch_size=64,
                verbose=1,
                class_weight=class_weight,
                callbacks=[sc_callback, cp_callback, csv_callback])

    # Do plot
    MakePlot(history)

    #
    df_log = pd.read_csv(os.path.join(pathCP, 'training.log'))
    df_log = df_log.nlargest(n=10, columns='val_auc')
    idx = df_log['epoch'].to_numpy() + 1

    yTest = rankdata(model.predict(xTest)) / len(df_test.index)

    for epoch in idx:
        print(epoch)
        stre = str(epoch)
        ncp = stre.zfill(4)
        checkpoint_path = os.path.join(pathCP, 'cp-' + ncp + '.ckpt')
        model.load_weights(checkpoint_path)
        yTest += rankdata(model.predict(xTest)) / len(df_test.index)

    df_test['target'] = yTest
    df_test.to_csv('rankall.csv', index=False)