import os
import shutil
import pandas as pd

# Copies files from the original folders into folders that are easily usable by flow_from_directory() keras method
# foldNUm specifies which k-fold ot use as the validation set.

if os.getcwd() == '/content/siim-isic':
    pathData = '/content/drive/My Drive/datasets/siim-isic/512x512-dataset-melanoma/512x512-dataset-melanoma'
    pathOut = '/content/drive/My Drive/datasets/siim-isic/'
    dfFold = pd.read_csv('/content/drive/My Drive/datasets/siim-isic/folds_08062020.csv')
else:
    pathData = './512x512-dataset-melanoma/512x512-dataset-melanoma'
    pathOut = os.path.join(os.getcwd(), 'data_split')
    dfFold = pd.read_csv(f'./folds_13062020.csv')

numFolds = 5

for idxFold in range(numFolds):

    print('Moving images...')

    for idx in range(len(dfFold)):
        if idx % 100 == 0:
            print(idx)
        fileName = dfFold.at[idx, 'image_id'] + '.jpg'
        fileIn = os.path.join(pathData, fileName)
        target = str(dfFold.at[idx, 'target'])

        if True: # dfFold.at[idx, 'source'] == 'ISIC20':
            if dfFold.at[idx, 'fold'] == idxFold:
                pathSplit = os.path.join(pathOut, 'data' + str(idxFold), 'val', target)
                if not os.path.isdir(pathSplit):
                    os.makedirs(pathSplit)
                fileOut = os.path.join(pathSplit, fileName)
                shutil.copyfile(fileIn, fileOut)

            else:
                pathSplit = os.path.join(pathOut, 'data' + str(idxFold), 'train', target)
                if not os.path.isdir(pathSplit):
                    os.makedirs(pathSplit)
                fileOut = os.path.join(pathSplit, fileName)
                shutil.copyfile(fileIn, fileOut)
