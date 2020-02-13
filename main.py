## Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed


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


def preprocess(data):
    #TODO Select boundary for artists (minimal 200 paintings?)
    data = data.sort_values(by=['paintings'], ascending=False)
    data = data[['name', 'paintings']]
    top_artists = data[data['paintings'] >= 200].reset_index()
    print(top_artists)

    # Make histogram
    artist_hist(top_artists)

    return top_artists


def gather_data():
    im_dir = './images/images'

    dat = pd.read_csv('artists.csv')
    dat = preprocess(dat)

    #TODO: Make function that greps the images

    check_directories(dat, im_dir)
    print_random_sample(dat, im_dir)
    return dat


#TODO: Build model so that a performance can be calculated.
def build_model(data):
    batch_size = 16
    train_input_shape = (224, 224, 3)



def main():
    dat = gather_data()
    preprocess(dat)
    build_model(dat)

main()