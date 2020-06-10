import os
import shutil
import pandas as pd

# Copies files from the original folders into folders that are easily usable by flow_from_directory() keras method

if os.getcwd() == '/content/siim-isic':
    pathData = '/content/drive/My Drive/datasets/siim-isic/512x512-dataset-melanoma/512x512-dataset-melanoma'
    pathOut = '/content/drive/My Drive/datasets/siim-isic/'
    dfFold = pd.read_csv('/content/drive/My Drive/datasets/siim-isic/folds_08062020.csv')
else:
    pathData = './512x512-dataset-melanoma/512x512-dataset-melanoma'
    pathOut = os.getcwd()
    dfFold = pd.read_csv(f'./folds_08062020.csv')

foldNum = 2
subSetFact = 1 # If on google colab, reduce data by a factor of this number

print('Moving images...')

c1 = 1
c2 = 1

for idx in range(len(dfFold)):
    if idx % 100 == 0:
        print(idx)
    fileName = dfFold.at[idx, 'image_id'] + '.jpg'
    fileIn = os.path.join(pathData, fileName)
    target = str(dfFold.at[idx, 'target'])

    if dfFold.at[idx, 'fold'] == foldNum:
        pathSplit = os.path.join(pathOut, 'data', 'val', target)
        if not os.path.isdir(pathSplit):
            os.makedirs(pathSplit)
        fileOut = os.path.join(pathSplit, fileName)
        if c1 == subSetFact:
            shutil.copyfile(fileIn, fileOut)
            c1 = 0
        c1 += 1
    else:
        pathSplit = os.path.join(pathOut, 'data', 'train', target)
        if not os.path.isdir(pathSplit):
            os.makedirs(pathSplit)
        fileOut = os.path.join(pathSplit, fileName)
        if c2 == subSetFact:
            shutil.copyfile(fileIn, fileOut)
            c2 = 0
        c2 += 1