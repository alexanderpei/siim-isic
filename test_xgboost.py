import os
import glob
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from sklearn.preprocessing import normalize

# Path which contains all of the forward propagations of all of the images through each of the networks.

pathIn = os.path.join(os.getcwd(), 'forward')
nModels = len(os.listdir(pathIn))
print(nModels)

foldNum = 0

# Reading the training data and test data and renaming the column for easier merge
df_x = pd.read_csv('folds_13062020.csv')
df_x = df_x.rename(columns={'image_id': 'image_name'})
nTrain = df_x.shape[0]

df_xTest = pd.read_csv('test.csv')
nTest = df_xTest.shape[0]

# This loop will go through every cp folder inside the path 'forward'. Each of these folders represents a different
# model or a different k-fold. For every test and train file inside each checkpoint folder, the predictions are
# averaged. Rankdata is optional, however it may help with comparisons between models. I'm not sure though.
# After looping through every file in a checkpoint, the data frame is merged into the top dataframe above.
# This merging fixes in case any of the files are not sorted by image name.
c = 0
for path in os.listdir(pathIn):

    tempTrain = np.zeros((60487))
    tempTest = np.zeros((nTest))

    for file in glob.glob(os.path.join(pathIn, path, 'train*')):
        df_train = pd.read_csv(file)
        #tempTrain += rankdata(df_train['target'].to_numpy()) / len(glob.glob(os.path.join(pathIn, path, 'train*'))) / len(df_train.index)
        tempTrain += df_train['target'].to_numpy() / len(glob.glob(os.path.join(pathIn, path, 'train*')))
    for file in glob.glob(os.path.join(pathIn, path, 'test*')):
        df_test = pd.read_csv(file)
        # tempTest += rankdata(df_test['target'].to_numpy()) / len(glob.glob(os.path.join(pathIn, path, 'test*'))) / len(df_test.index)
        tempTest += df_test['target'].to_numpy() / len(glob.glob(os.path.join(pathIn, path, 'test*')))

    df_train['target'] = tempTrain
    df_test['target'] = tempTest
    c += 1
    df_x = df_x.merge(df_train, how='left', on='image_name', suffixes=(str(c), str(c) + 'y'))
    df_xTest = df_xTest.merge(df_test, how='left', on='image_name', suffixes=(str(c), str(c) + 'y'))

# Getting the indexes of the training and validation set based on which foldNum you want to use.
idxVal = df_x.index[df_x['fold'] == foldNum]
idxTrain = df_x.index[df_x['fold'] != foldNum]

# Gathering the training data for the images. We index into the last nModels (note the counter in the loops above)
# because the rest of the array contains junk like the source and name etc.
xIm = df_x.to_numpy()
xIm = xIm[:, -nModels:].astype(np.float32)\

# The training labels
y = df_x['target1'].to_numpy().astype(np.float32)

# Test data
xImTest = df_xTest.to_numpy()
xImTest = xImTest[:, -nModels:].astype(np.float32)

# Load in the meta data

def onehot(df):
    # This function will load in the meta data using a one hot encoding method. Values that are missing have their
    # own one hot encoding.
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


# One hot encode the pandas dataframe for the training and test metadata
xMeta = onehot(pd.read_csv('folds_13062020.csv'))
xMetaTest = onehot(pd.read_csv('test.csv'))

# Concatenate the image and meta data
x = np.hstack((xIm, xMeta))
xTest = np.hstack((xImTest, xMetaTest))

# Split into training and validation set
xTrain = x[idxTrain, :]
xVal = x[idxVal, :]

yTrain = y[idxTrain]
yVal = y[idxVal]

model = XGBClassifier(n_estimators=2000,
                        max_depth=15,
                        objective='multi:softprob',
                        seed=0,
                        nthread=-1,
                        learning_rate=0.15,
                        num_class = 2,
                        scale_pos_weight = (32542/584))

model.fit(xTrain, yTrain)

y_pred = model.predict_proba(xVal)[:,1]
auc = roc_auc_score(yVal, y_pred)
print("AUC: %.2f%%" % (auc))

y_pred = model.predict_proba(xTest)[:,1]
df_test['target'] = y_pred
df_test.to_csv('xgboost.csv', index=False)
