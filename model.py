import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D,BatchNormalization
import GeneratorClass

def  preProcessingForGenerator(train_dict):
    input_ids = [x for x,_ in train_dict.items()]
    output_ids = [x for _,x in train_dict.items()]

    train_generator = GeneratorClass.DataGenerator(input_ids,output_ids,train_dict)
    print(type(train_generator))
    return train_generator

def create_model(train_generator):
    model = Sequential()
    in_shape = (300,300,3)

    model.add(Conv2D(64,(3,3),activation = 'relu',input_shape = in_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))

    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])

    model.fit_generator(generator = train_generator,
                        epochs = 10,
                        steps_per_epoch = 10,
                        verbose = 1)
