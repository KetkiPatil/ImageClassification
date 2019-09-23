import numpy as np
import os
from os import path
import pandas as pd
import cv2
import random
import GeneratorClass

new_file = './Train/new_train_data.txt'

def shuffle_data(train_data):

    with open(train_data,'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()

    with open(new_file,'w+') as target:
        for _, line in data:
            target.write(line)
    target.close()
    source.close()

def create_files(train_data):
    train_dict = dict()
    labels = {'A':1, 'B':2, 'C':3, 'D':4}
    train_coordinates = dict()
    with open(new_file,'r') as f:
        f1 = f.readlines()

    for line in f1:

        coordinates = []

        image_data = line.split(",")
        image_name = image_data[0][0:10]+".jpg"
        image_label = image_data[5][0]

        train_dict[image_name] = labels[image_label]

        coordinates.append(image_data[1])
        coordinates.append(image_data[2])
        coordinates.append(image_data[3])
        coordinates.append(image_data[4])

        train_coordinates[image_name] = coordinates

    return train_dict,train_coordinates
