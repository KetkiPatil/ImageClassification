import os
from os import path
import preProcess
import GeneratorClass
import model
from keras.models import Model

main_dir = '../Bizlers/'
train_dir = './Train'
train_data = os.path.join(train_dir,"train.txt")
test_data = './Test/'

if __name__ == '__main__':
    preProcess.shuffle_data(train_data)

    train_dict,train_coordinates=preProcess.create_files(train_data)

    train_generator = model.preProcessingForGenerator(train_dict)

    model.create_model(train_generator)
