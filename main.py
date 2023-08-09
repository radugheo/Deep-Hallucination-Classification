import tensorflow as tf
import pandas as pd
import numpy as np
import os
import shutil
from matplotlib import pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import seaborn as sns

######################################## REARRANGE FILES ########################################
# aceste functii muta imaginile de instruire si de validare in subdirectoarele specifice claselor lor, facilitand incarcarea datelor cu 'image_dataset_from_directory'

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

# classifyTrain()
# classifyVal()

######################################## TAKE DATA ########################################
# incarcam seturile de date de instruire si de validare ca TensorFlow Dataset-uri, redimensionand imaginile la dimensiuni de (imgHeight, imgWidth)

imgHeight = 64
imgWidth = 64
trainDataset = tf.keras.preprocessing.image_dataset_from_directory(
    'train_images/',
    seed=123,
    image_size=(imgHeight, imgWidth),
    label_mode='categorical')
validationDataset = tf.keras.preprocessing.image_dataset_from_directory(
    'val_images/',
    seed=123,
    image_size=(imgHeight, imgWidth),
    label_mode='categorical')

######################################## TUNING ########################################
# setam callback-urile pentru antrenare:
# - ReduceLROnPlateau (pentru a reduce rata de invatare cand modelul nu se imbunatateste)
# - EarlyStopping (pentru a opri antrenarea cand modelul nu se imbunatateste)
# - ModelCheckpoint (pentru a salva cel mai bun model)
# aplicam augmentarea datelor prin inversarea imaginilor pe orizontala si concatenam cu ce aveam pana acum
# amestecam noile date de antrenament si le preprocesam pentru a fi gata pentru model

def callbacksSetup(checkpoint):
    reduceLearningRate = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, mode='min', factor=0.5, min_lr=1e-5)
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=75, verbose=1, mode='max', restore_best_weights=True)
    checkpointSave = ModelCheckpoint(checkpoint, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    return [reduceLearningRate, earlyStopping, checkpointSave]
def deterministicFlipDataAugmentation(image, label):
    imageFlipped = tf.image.flip_left_right(image)
    return imageFlipped, label

augmentedDataset = trainDataset.map(deterministicFlipDataAugmentation)
finalDataset = trainDataset.concatenate(augmentedDataset)
classNames = trainDataset.class_names
finalDataset, validationDataset = finalDataset.cache().shuffle(24000).prefetch(buffer_size=tf.data.AUTOTUNE), validationDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomZoom(-.5, input_shape=(imgHeight, imgWidth, 3), fill_mode="wrap"),
    ]
) # zoom aleator la imagini

######################################## TRAIN MODEL ########################################
# cream arhitectura modelului nostru CNN
# folosim Dropout pentru a preveni overfitting-ul si BatchNormalization pentru a normaliza intrarile fiecarui strat, ca sa imbunatatim performanta si stabilitatea modelului

model092 = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(imgHeight, imgWidth, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(96, activation='softmax')
])
# compilam modelul, folosind optimizatorul adam si ca functie de pierdere, categorical_crossentropy
model092.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 250
# antrenam modelul pentru 250 de epoci, salvand rezultatele in 'history' ca sa vedem ulterior progresul modelului
history = model092.fit(trainDataset, validation_data=validationDataset, callbacks=callbacksSetup('model092.keras'), epochs=epochs)

####################################### PLOT RESULTS ########################################
# functii pentru a trasa grafice de precizie si pierdere de antrenare si validare

def plotResults():
    trainAccuracy, validationAccuracy, trainLoss, validationLoss = history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss']
    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), trainAccuracy, label='Training Accuracy')
    plt.plot(range(epochs), validationAccuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), trainLoss, label='Training Loss')
    plt.plot(range(epochs), validationLoss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()
# plotResults()

######################################## PLOT CONFUSION MATRIX ########################################
# functie pentru a trasa matricea de confuzie, care ne arata cate exemple au fost corect clasificate si cate au fost clasificate gresit

def plotConfusionMatrix():
    validationImages, validationLabels = [], []
    for image, label in validationDataset:
        validationImages.append(image)
        validationLabels.append(label)
    validationImages, validationLabels = np.concatenate(validationImages), np.concatenate(validationLabels)
    validationPredictions = model092.predict(validationImages)
    # etichetele sunt în format 'one-hot', trebuie sa le transformam inapoi in formatul original
    validationPredictedClasses, validationLabelsClasses = np.argmax(validationPredictions, axis=1), np.argmax(validationLabels, axis=1)
    confusionMatrix = confusion_matrix(validationLabelsClasses, validationPredictedClasses)
    plt.figure(figsize=(20, 20))
    sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='viridis', xticklabels=classNames, yticklabels=classNames, cbar_kws={"shrink": .5})
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.rcParams.update({'font.size': 6})
    plt.show()
# plotConfusionMatrix()

######################################## PREDICT LABELS ########################################
# facem predictii pentru setul de date de test
# pentru fiecare imagine, o incarcam, o preprocesam, facem predictia cu modelul antrenat mai sus
# apoi adaugam numele imaginii si clasa prezisa in fisierul de output

testFileInput = pd.read_csv('test.csv')
imageFileNames = testFileInput['Image'].values
fileOutput = open('submission.csv', 'w')
print('Image,Class', file=fileOutput)
for i in range(len(imageFileNames)):
    loadedImage = tf.keras.utils.load_img('test_images/' + imageFileNames[i], target_size=(imgHeight, imgWidth))
    loadedImageArray = tf.keras.utils.img_to_array(loadedImage)
    loadedImageArray = tf.expand_dims(loadedImageArray, 0)
    loadedImagePrediction = model092.predict(loadedImageArray)
    predictedScore = tf.nn.softmax(loadedImagePrediction[0])
    print(f'{imageFileNames[i]},{classNames[np.argmax(predictedScore)]}', file=fileOutput)