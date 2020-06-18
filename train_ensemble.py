import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from focal_loss import focal_loss

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import normalize

from HelperFunctions import MakePlot

# Path which contains all of the forward propagations of all of the images through each of the networks.

pathIn = os.path.join(os.getcwd(), 'forward')
foldList = os.listdir(pathIn)
nModels = len(foldList)

train = True
foldNum = 0

df_fold = pd.read_csv('folds_13062020.csv')
df_fold = df_fold.rename(columns={'image_id': 'image_name'})

df = pd.read_csv(r'C:\Users\Alex\PycharmProjects\siim-isic\forward\cp_1_0\train_0_0.csv')
nTrain = df.shape[0]

df_testall = pd.read_csv(r'C:\Users\Alex\PycharmProjects\siim-isic\forward\cp_1_0\test_0_0.csv')
nTest = df_testall.shape[0]

c = 0
for path in os.listdir(pathIn):

    tempTrain = np.zeros((nTrain))
    tempTest = np.zeros((nTest))

    for file in glob.glob(os.path.join(pathIn, path, 'train*')):
        df_train = pd.read_csv(file)
        tempTrain += df_train['target'].to_numpy() / len(glob.glob(os.path.join(pathIn, path, 'train*')))

    for file in glob.glob(os.path.join(pathIn, path, 'test*')):
        df_test = pd.read_csv(file)
        tempTest += df_test['target'].to_numpy() / len(glob.glob(os.path.join(pathIn, path, 'test*')))

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

# xVal = normalize(xVal, axis=0)
# xTrain = normalize(xTrain, axis=0)
# xTest = normalize(xTest, axis=0)

if train:
    # Model Params

    lr = 0.0001
    nEpoch = 100
    opt = tf.keras.optimizers.Adam(
        lr=lr)
    opt = tfa.optimizers.Lookahead(opt)

    loss = [focal_loss(alpha=0.75, gamma=2)]

    # Callback

    sc_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=5,
        min_lr=1e-6)

    # Train

    model = Sequential()
    model.add(Dense(20, input_dim=nModels, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
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
                callbacks=[sc_callback])

    # Do plot
    MakePlot(history)

    yTest = model.predict(xTest)

    df_test['target'] = yTest
    df_test.to_csv('test_ensemble.csv', index=False)