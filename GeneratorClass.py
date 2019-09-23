import keras
import numpy as np
import os
import ImageClassification
from keras.models import Model
#from ImageClassification import train_coordinates

def preProcessImage(filename):
    if path.exists(filename):
        img = cv2.imread(filename)
        coordinates = ImageClassification.train_coordinates[filename]
        #crop_img = img[int(image_data[2]):int(image_data[4]), int(image_data[1]):int(image_data[3])]
        crop_img = img[int(coordinates[2]):int(coordinates[4]), int(coordinates[1]):int(coordinates[3])]
        resize_img  = cv2.resize(crop_img, dsize=(300,300))
        return resize_img

class DataGenerator(keras.utils.Sequence):
    def __init__ (self, input_ids, output_ids, input_dict, batch_size = 4,out_dims = (300,300),shuffle=True):
        self.out_dims = out_dims
        self.batch_size = batch_size
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.input_dict = input_dict
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.input_ids))/self.batch_size)

    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item    

    def __getitem(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.input_ids[k] for k in indexes]
        X, y = self.__data_generation(list_ids_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.input_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        X = np.empty((self.batch_size))
        y = np.empty((self.batch_size))

        for i, ID in enumerate(list_ids_temp):
            image = preprocessImage(ID)
            X[i,] = image
            y[i,] = self.input_dict[ID]

        return X,y
