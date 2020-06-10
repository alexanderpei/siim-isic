import os
import shutil
import pandas as pd

# Copies files from the original folders into folders that are easily usable by flow_from_directory() keras method

if os.getcwd() == '/content/siim-isic':
    pathData = '/content/drive/My Drive/KaggleData/dataset-siim-isic/512x512-dataset-melanoma/512x512-dataset-melanoma'
    dfFold = pd.read_csv('/content/drive/My Drive/KaggleData/dataset-siim-isic/folds.csv')
else:
    pathData = './512x512-dataset-melanoma/512x512-dataset-melanoma'
    dfFold = pd.read_csv(f'./folds.csv')

foldNum = 2

print('Moving images...')

for idx in range(len(dfFold)):
    fileName = dfFold.at[idx, 'image_id'] + '.jpg'
    fileIn = os.path.join(pathData, fileName)
    target = str(dfFold.at[idx, 'target'])

    if dfFold.at[idx, 'fold'] == foldNum:
        pathOut = os.path.join(os.getcwd(), 'data', 'val', target)
    else:
        pathOut = os.path.join(os.getcwd(), 'data', 'train', target)

    if not os.path.isdir(pathOut):
        os.makedirs(pathOut)

    fileOut = os.path.join(pathOut, fileName)
    shutil.copyfile(fileIn, fileOut)
