## Import libraries

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import pickle

import tensorflow as tf
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.applications import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.initializers import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing.image import array_to_img

from sklearn.model_selection import train_test_split

from pathlib import Path

from numpy.random import seed

from tensorflow import set_random_seed
from PIL import Image
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# import tensorflow.python.keras


def check_directories(data, im_dir):
    # Explore images of top artists
    dir_names = data['name'].str.replace(' ', '_').values

    # See if all directories exist
    for name in dir_names:
        if os.path.exists(os.path.join(im_dir, name)):
            print("Found -->", os.path.join(im_dir, name))
        else:
            print("Did not find -->", os.path.join(im_dir, name))
    return

## Create histogram of artists and their paintings
def artist_hist(data):
    artistlist = data.name.unique()
    hist = [0] * len(data)
    index = 0

    for artist in artistlist:
        temp = data.loc[data['name'] == artist]
        hist[index] = temp.paintings.sum()
        index += 1
    print(hist)
    numOfPaintings = pd.DataFrame([hist], columns= artistlist.tolist())
    numOfPaintings.plot(kind='bar')
    plt.show()

def print_random_sample(data, im_dir):
    ##TODO rewrite to own code (Taken directly from kaggle)
    n = 5
    fig, axes = plt.subplots(1, n, figsize=(20, 10))

    for i in range(n):
        selected_artist = random.choice(data['name'].str.replace(' ', '_').values)
        selected_image = random.choice(os.listdir(os.path.join(im_dir, selected_artist)))
        image_file = os.path.join(im_dir, selected_artist, selected_image)
        image = plt.imread(image_file)
        axes[i].imshow(image)
        axes[i].set_title("Artist: " + selected_artist.replace('_', ' '))
        axes[i].axis('off')

    plt.show()


def exploratory(data):
    artist_hist(data)

def print_img(paths):
    # load the image
    img = load_img(paths[0])

    img2 = img.resize((224,224))

    # convert to numpy array
    img_array = img_to_array(img2)
    print(img_array.dtype)
    print(img_array.shape)

    # convert back to image
    img_pil = array_to_img(img_array)
    print(type(img))

    fig, axes = plt.subplots(1, 2, figsize=(20,10))

    # Original image
    image = img
    axes[0].imshow(image)
    axes[0].set_title("An original Image")
    axes[0].axis('off')

    # Transformed image
    aug_image = img2
    axes[1].imshow(aug_image)
    axes[1].set_title("A transformed Image")
    axes[1].axis('off')

    plt.show()

def preprocess(data):
    #TODO Select boundary for artists (minimal 200 paintings?)
    data = data.sort_values(by=['paintings'], ascending=False)
    data = data[['name', 'paintings']]
    top_artists = data[data['paintings'] >= 200].reset_index()
    print(top_artists)
    # Make histogram
    # artist_hist(top_artists)

    return top_artists


def gather_data():
    im_dir = './images/images'

    dat = pd.read_csv('artists.csv')
    dat = preprocess(dat)

    check_directories(dat, im_dir)
    # print_random_sample(dat, im_dir)
    return dat

def make_ext(paths: Path, label: str) -> list:
    return [(x, label) for x in paths.glob('*' + '.jpg')]

def create_paths(dat):
    paths = []
    artist_dir = dat['name'].str.replace(' ', '_').values
    DATA_DIR = Path('./images/images/')

    for artist in artist_dir:
        paths += make_ext(DATA_DIR / artist, artist)

    return paths

def read_img(paths):
    x_train = np.array([])
    y_train = np.array([]) #["" for x in range(len(paths))]
    i=0

    for p in paths:
        i+=1
        img = load_img(p[0])
        img = img.resize((224,224))
        img = img_to_array(img)
        x_train = np.append(x_train, img)
        y_train = np.append(y_train, p[1])
        if(i % 200 == 0):
            print(i)
        elif(i > 3000 and i % 10 == 0):
            print(i)

    # print(x_train)
    # print(y_train)

    np.save('labels_NAMES',y_train)

    labelizer = LabelEncoder()
    y_train = labelizer.fit_transform(y_train)
    y_train = to_categorical(y_train)

    
    np.save('paintings', x_train)
    np.save('labels', y_train)

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    
    axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')
    axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend(loc='best')

    axes[1].plot(epochs, loss, 'r-', label='Training Loss')
    axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend(loc='best')
    
    plt.show()

def save_training(h, testName):

    acc = h.history['acc']
    val_acc = h.history['val_acc']
    loss = h.history['loss']
    val_loss = h.history['val_loss']
    epochs = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    
    axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')
    axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend(loc='best')

    axes[1].plot(epochs, loss, 'r-', label='Training Loss')
    axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend(loc='best')

    plt.savefig(testName)
    np.save(testName, [acc, val_acc, loss, val_loss, epochs])


def build_model(train_generator,STEP_SIZE_TRAIN,valid_generator,STEP_SIZE_VALID,testName):

    batch_size = 16
    input_shape = (224, 224, 3)
    num_classes = 10
    n_epoch = 50

    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                              verbose=1, mode='auto')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if(testName == 'Dropout2' or testName == 'Dropout2Weight'):
        model.add(Dropout(0.5))
    if(testName == 'Weight' or testName == 'Weight2' or testName == 'DropoutWeight' or testName == 'DropoutWeight2'):
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    else:
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    if(testName == 'Dropout' or testName == 'Dropout2' or testName == 'DropoutBatch' or testName == 'DropoutWeight2' or testName == 'Dropout2Weight'):
        model.add(Dropout(0.5))
    if(testName == 'Batch' or testName == 'DropoutBatch'):
        model.add(BatchNormalization())
    model.add(Flatten())
    if(testName == 'Weight2' or testName == 'DropoutWeight2'):
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer=adam(), metrics=['accuracy'])
    print(model.summary())

    optimizer = Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])

    history = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                              epochs=n_epoch,
                              shuffle=True,
                              verbose=1,
                              callbacks=[reduce_lr, early_stop],
                              use_multiprocessing=True,
                              workers=8
                              )
    
    # plot_training(history)
    save_training(history, testName)

def initModel(t):
    batch_size = 16
    train_input_shape = (224, 224, 3)
    n_classes = 10

    images_dir = './images/images/'
    artists = ['Vincent_van_Gogh', 'Edgar_Degas', 'Pablo_Picasso', 'Pierre-Auguste_Renoir', 'Paul_Gauguin', 'Francisco_Goya', 'Rembrandt', 'Alfred_Sisley', 'Titian', 'Marc_Chagall']
    
    
    

    train_datagen = ImageDataGenerator(validation_split=0.2,
                                    rescale=1./255.,
                                    #rotation_range=45,
                                    #width_shift_range=0.5,
                                    #height_shift_range=0.5,
                                    shear_range=5,
                                    #zoom_range=0.7,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    )

    train_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                        class_mode='categorical',
                                                        target_size=train_input_shape[0:2],
                                                        batch_size=batch_size,
                                                        subset="training",
                                                        shuffle=True,
                                                        classes=artists
                                                    )

    valid_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                        class_mode='categorical',
                                                        target_size=train_input_shape[0:2],
                                                        batch_size=batch_size,
                                                        subset="validation",
                                                        shuffle=True,
                                                        classes=artists
                                                    )

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
    print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)

    build_model(train_generator,STEP_SIZE_TRAIN,valid_generator,STEP_SIZE_VALID,t)

def draw_acc():
    base = np.load('./tests/Baseline.npy')
    drop = np.load('./tests/Dropout.npy')
    wei = np.load('./tests/Weight.npy')
    batch = np.load('./tests/Batch.npy')
    drop2wei = np.load('./tests/Dropout2Weight.npy')
    dropBatch = np.load('./tests/DropoutBatch.npy')
    dropwei = np.load('./tests/DropoutWeight.npy')

    acc = [base[0], drop[0], wei[0], batch[0], drop2wei[0], dropBatch[0], dropwei[0]]
    val_acc = [base[1], drop[1], wei[1], batch[1], drop2wei[1], dropBatch[1], dropwei[1]]
    loss = [base[2], drop[2], wei[2], batch[2], drop2wei[2], dropBatch[2], dropwei[2]]
    val_loss = [base[3], drop[3], wei[3], batch[3], drop2wei[3], dropBatch[3], dropwei[3]]
    names = ['Baseline', 'Dropout', 'Weight Decay', 'Batch Normalization', 'Dropout2Weight', 'DropoutBatch', 'DropoutWeight']

    col = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    i=0
    j=0

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    for ac in acc:
        axes[0].plot(range(len(ac)), ac, col[i], label=(names[i]))
        i+=1
    
 
    for val_ac in val_acc:
        axes[1].plot(range(len(val_ac)), val_ac, col[j], label=(names[j]))
        j+=1

    axes[0].set_title('Training Accuracy')
    axes[0].legend(loc='best')

    axes[1].set_title('Validation Accuracy')
    axes[1].legend(loc='best')

def draw_loss():
    base = np.load('./tests/Baseline.npy')
    drop = np.load('./tests/Dropout.npy')
    wei = np.load('./tests/Weight.npy')
    batch = np.load('./tests/Batch.npy')
    drop2wei = np.load('./tests/Dropout2Weight.npy')
    dropBatch = np.load('./tests/DropoutBatch.npy')
    dropwei = np.load('./tests/DropoutWeight.npy')

    acc = [base[0], drop[0], wei[0], batch[0], drop2wei[0], dropBatch[0], dropwei[0]]
    val_acc = [base[1], drop[1], wei[1], batch[1], drop2wei[1], dropBatch[1], dropwei[1]]
    loss = [base[2], drop[2], wei[2], batch[2], drop2wei[2], dropBatch[2], dropwei[2]]
    val_loss = [base[3], drop[3], wei[3], batch[3], drop2wei[3], dropBatch[3], dropwei[3]]
    names = ['Baseline', 'Dropout', 'Weight Decay', 'Batch Normalization', 'Dropout2Weight', 'DropoutBatch', 'DropoutWeight']

    col = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    i=0
    j=0

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    for los in loss:
        axes[0].plot(range(len(los)), los, col[i], label=(names[i]))
        i+=1
    
    j=0
    for val_los in val_loss:
        axes[1].plot(range(len(val_los)), val_los, col[j], label=(names[j]))
        j+=1

    # axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
    axes[0].set_title('Training Loss')
    axes[0].legend(loc='best')

    # axes[1].plot(epochs, loss, 'r-', label='Training Loss')
    # axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend(loc='best')

    plt.show()

def main():

    # dat = gather_data()
    # build_model(dat)
    
    tests = ['Baseline','Dropout', 'Dropout2' 'Weight', 'Weight2' 'Batch', 'Dropout2Weight', 'DropoutBatch']

    for test in tests:
        initModel(test)

main()