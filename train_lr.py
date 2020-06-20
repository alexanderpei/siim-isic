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

pathOut = os.path.join(os.getcwd(), 'forward', 'lr')
if not os.path.isdir(pathOut):
    os.makedirs(pathOut)

def onehot(df):
    gender = np.zeros((df.shape[0], 3))
    for i in range(df.shape[0]):
        if df.at[i, 'sex'] == 'female':
            gender[i, 0] = 1
        elif df.at[i, 'sex'] == 'male':
            gender[i, 1] = 1
        else:
            gender[i, 2] = 1

    uniq = ['oral/genital', 'head/neck', 'unknown', 'upper extremity', 'palms/soles', 'lower extremity', 'lateral torso', 'torso']

    site = np.zeros((df.shape[0], len(uniq)))
    for i in range(df.shape[0]):
        try:
            idx = uniq.index(df.at[i, 'anatom_site_general_challenge'])
        except ValueError:
            idx = 3
        site[i, idx] = 1

    age = np.zeros((df.shape[0], 1))
    for i in range(df.shape[0]):
        age[i] = df.at[i, 'age_approx']
        if np.isnan(age[i]):
            age[i] = age[i-1]

    age = normalize(age)

    X = np.concatenate((gender, site, age), axis=1)
    return X.astype(np.float32)


# One hot encode the pandas dataframe
X = onehot(pd.read_csv('folds_13062020.csv'))

# Split based on folds
df_fold = pd.read_csv('folds_13062020.csv')
df_fold = df_fold.rename(columns={'image_id': 'image_name'})
foldNum = 0

idxVal = df_fold.index[df_fold['fold'] == foldNum]
idxTrain = df_fold.index[df_fold['fold'] != foldNum]

xVal = X[idxVal, :]
xTrain = X[idxTrain, :]

yVal = df_fold['target'].loc[idxVal].to_numpy()
yTrain = df_fold['target'].loc[idxTrain].to_numpy()

# Load in the test data
df_testall = pd.read_csv('test.csv')
xTest = onehot(df_testall)

# Model

lr = 0.0001
nEpoch = 30
opt = tf.keras.optimizers.Adam(
    lr=lr)
opt = tfa.optimizers.Lookahead(opt)

loss = [focal_loss(alpha=0.75, gamma=2)]
loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)

# Can try using class weights to fix bias in the data. Down-weighting the benign class since there are more of them.
class_weight = {0: 1, 1: 1}
# Callback
# Callback

sc_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.5,
    patience=5,
    min_lr=1e-6)

# Train

model = Sequential()
model.add(Dense(20, input_dim=xTrain.shape[1], activation='relu'))
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
    class_weight=class_weight,
    verbose=1,
    callbacks=[sc_callback])

# Do plot
MakePlot(history)

#

print(xTest.shape)
print(X.shape)

df_train = pd.read_csv('marking.csv')
df_train = df_train.rename(columns={'image_id': 'image_name'})

X = onehot(df_train)

yTrain = model.predict(X)
yTest = model.predict(xTest)

df_train['target'] = yTrain
df_testall['target'] = yTest

df_train.to_csv(os.path.join(pathOut, 'train_2.csv'), columns=['image_name', 'target'], index=False)
df_testall.to_csv(os.path.join(pathOut, 'test_2.csv'), columns=['image_name', 'target'], index=False)