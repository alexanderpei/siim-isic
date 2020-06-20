import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import rankdata

pathIn = os.path.join(os.getcwd(), 'forward')

df_testall = pd.read_csv(r'C:\Users\Alex\PycharmProjects\siim-isic\forward\cp_3_0\test_0_0.csv')
nTest = df_testall.shape[0]

tempTest = np.zeros((nTest))

c = 0
for path in os.listdir(pathIn):
    for file in glob.glob(os.path.join(pathIn, path, 'test*')):
        df = pd.read_csv(file)
        tempTest += rankdata(df['target'].to_numpy()) / len(glob.glob(os.path.join(pathIn, path, 'test*'))) / len(df.index)
        c += 1

df['target'] = tempTest
print(c)
df.to_csv('halloq.csv', index=False)