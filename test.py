import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

df = pd.read_csv('marking.csv')

print(df)

gender = np.zeros((df.shape[0], 3))

print(gender)

for i in range(df.shape[0]):

    if df.at[i, 'sex'] == 'female':
        gender[i, 0] = 1
    elif df.at[i, 'sex'] == 'male':
        gender[i, 1] = 1
    else:
        gender[i, 2] = 1

print(gender)

x = df['anatom_site_general_challenge'].tolist()
uniq = list(set(x))
print(uniq)

site = np.zeros((df.shape[0], len(uniq)))

for i in range(df.shape[0]):

    idx = uniq.index(df.at[i, 'anatom_site_general_challenge'])
    site[i, idx] = 1

print(site.shape)
print(gender.shape)

X = np.concatenate((gender, site), axis=1)
y = df['target'].to_numpy()