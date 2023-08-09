import os
import shutil
import pandas as pd

def classifyTrain():
    trainFileInput = pd.read_csv("train.csv")
    for index, row in trainFileInput.iterrows():
        destinationDirectory = os.path.join('train_images/', str(row['Class']))
        if not os.path.exists(destinationDirectory):
            os.makedirs(destinationDirectory)
        sourceFile = os.path.join('train_images/', row['Image'])
        destinationFile = os.path.join(destinationDirectory, row['Image'])
        shutil.move(sourceFile, destinationFile)
def classifyVal():
    validationFileInput = pd.read_csv("val.csv")
    for index, row in validationFileInput.iterrows():
        destinationDirectory = os.path.join('val_images/', str(row['Class']))
        if not os.path.exists(destinationDirectory):
            os.makedirs(destinationDirectory)
        sourceFile = os.path.join('val_images/', row['Image'])
        destinationFile = os.path.join(destinationDirectory, row['Image'])
        shutil.move(sourceFile, destinationFile)

classifyTrain()
classifyVal()