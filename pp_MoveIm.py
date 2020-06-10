import os
import shutil
import pandas as pd

# Copies files from the original folders into folders that are easily usable by flow_from_directory() keras method

if os.getcwd() == '/content/siim-isic':
    pathData = '/content/drive/My Drive/KaggleData/dataset-siim-isic/512x512-dataset-melanoma/512x512-dataset-melanoma'
    pathOut = '/content/drive/My Drive/KaggleData/dataset-siim-isic/'
    dfFold = pd.read_csv('/content/drive/My Drive/KaggleData/dataset-siim-isic/folds08062020.csv')
else:
    pathData = './512x512-dataset-melanoma/512x512-dataset-melanoma'
    pathOut = os.getcwd()
    dfFold = pd.read_csv(f'./folds08062020.csv')

foldNum = 2

print('Moving images...')

for idx in range(len(dfFold)):
    fileName = dfFold.at[idx, 'image_id'] + '.jpg'
    fileIn = os.path.join(pathData, fileName)
    target = str(dfFold.at[idx, 'target'])

    if dfFold.at[idx, 'fold'] == foldNum:
        pathSplit = os.path.join(pathOut, 'data', 'val', target)
    else:
        pathSplit = os.path.join(pathOut, 'data', 'train', target)

    if not os.path.isdir(pathSplit):
        os.makedirs(pathSplit)

    fileOut = os.path.join(pathSplit, fileName)
    shutil.copyfile(fileIn, fileOut)
