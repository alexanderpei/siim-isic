import pandas as pd
import numpy as np
import os
from scipy.stats import rankdata

pathIn = r'C:\Users\Alex\Desktop\subs2'

all = np.zeros((10982))

c = 0
for root, dirs, files in os.walk(pathIn):
    for file in files:
        df = pd.read_csv(os.path.join(root, file))
        #all += rankdata(df['target'].to_numpy()) / len(df.index)
        all += df['target'].to_numpy()
        c += 1

print(c)
df['target'] = all/c
df.to_csv('test_ensemble.csv', index=False)
