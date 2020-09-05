# Import packages
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from bullet import Bullet, Check, YesNo, Input, SlidePrompt
from bullet import colors
import datetime
import csv
from tensorflow.keras.callbacks import TensorBoard
from cliparser import n_split, epochs, model_nom, pathData, optimizer, dropout1, dropout2, batch_size, learningrate, momentum, nesterov, beta1, beta2, amsgrad,tflite
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools


# add prompt line for model noun to crush it if it's already existent or exit if not
if model_nom + '.h5' in os.listdir('Models/'):
    cli = SlidePrompt(
        [
            YesNo("file already exist, do you want to overwrite it :",
                  word_color=colors.foreground["yellow"])])
    choice = cli.launch()
    if choice[0][1] == True:
        os.remove("Models/" + model_nom + ".h5")
    else:
        exit()

# create a folder of logs including models in orther to visualize them with tensorboard tool
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_nom))

# create path data
PATH = os.path.join(pathData)

# define training and validation paths
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# create a list 'train' containing paths to every class in train folder
train = []
for i in range(len(os.listdir(train_dir))):
    # directory with our training pictures
    train += [os.path.join(train_dir, os.listdir(train_dir)[i])]
    train = sorted(train)

# create a list 'validation' containing paths to every class in validation folder
validation = []
for i in range(len(os.listdir(validation_dir))):
    # directory with our validation pictures
    validation += [os.path.join(validation_dir,
                                os.listdir(validation_dir)[i])]
    validation = sorted(validation)

# Understand the data
total_train = 0
for i in range(len(os.listdir(train_dir))):
    # total_train = total number of images in all classes in train folder
    total_train += len(os.listdir(train[i]))

total_val = 0
for i in range(len(os.listdir(validation_dir))):
    # total_val =total number of images in all classes in validation folder
    total_val += len(os.listdir(validation[i]))

# define attributes
IMG_HEIGHT = 150
IMG_WIDTH = 150
RGB_channels = 3  # three colors: Red,Green & Blue
filters_for_CONV1 = 16  # the number of output filters in the convolution 1
filters_for_CONV2 = 32  # the number of output filters in the convolution 2
filters_for_CONV3 = 64  # the number of output filters in the convolution 3
Kernel_SIZE = 3  # An integer or tuple/list of 2 integers,
# specifying the height and width of the 2D convolution window.
# Can be a single integer to specify the same value for all spatial dimensions.
units1 = 512  # Positive integer, dimensionality of the output space
units2 = len(os.listdir(train_dir))
units3 = dropout1
units4 = dropout2

# Data preparation
# Generator for our training data
train_image_generator = ImageDataGenerator(rescale=1. / 255)

# Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

#  read the images from train folder
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(
                                                               IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

#  read the images from validation folder
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(
                                                                  IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

# When using any layer as the first layer in a model,
# provide the keyword argument input_shape (tuple of integers,
# does not include the sample axis), e.g. input_shape=(128, 128, 3)
# for 128x128 RGB pictures in data_format="channels_last".

# define convolutions & fully connected layers
model = Sequential([
    Conv2D(filters_for_CONV1, Kernel_SIZE, padding='same',
           activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, RGB_channels)),
    MaxPooling2D(),
    Conv2D(filters_for_CONV2, Kernel_SIZE, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(filters_for_CONV3, Kernel_SIZE, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(units3),
    Dense(units1, activation='relu'),
    Dropout(units4),
    Dense(units2, activation='softmax')
])

# switch beteween different optimizers
if optimizer == 'sgd':
    dynamic_optimizer = tf.keras.optimizers.SGD(
        learning_rate=learningrate, momentum=momentum, nesterov=nesterov)
elif optimizer == 'rmsprop':
    dynamic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learningrate)
elif optimizer == 'adagrad':
    dynamic_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learningrate)
elif optimizer == 'adadelta':
    dynamic_optimizer = tf.keras.optimizers.Adadelta(
        learning_rate=learningrate)
elif optimizer == 'adamax':
    dynamic_optimizer = tf.keras.optimizers.Adamax(
        learning_rate=learningrate, beta_1=beta1, beta_2=beta2)
elif optimizer == 'nadam':
    dynamic_optimizer = tf.keras.optimizers.Nadam(
        learning_rate=learningrate, beta_1=beta1, beta_2=beta2)
else:  # Adam
    dynamic_optimizer = tf.keras.optimizers.Adam(
        learning_rate=learningrate, beta_1=beta1, beta_2=beta2, amsgrad=amsgrad)

# build a model architecture with CNN
model.compile(optimizer=dynamic_optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Summarize Model
# Keras provides a way to summarize a model.
# The summary is textual and includes information about: The layers and their order in the model.
model.summary()

# train our model to get all the paramters to the correct value so that we can map our inputs to our outputs.
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[tensorboard]
)

test_steps_per_epoch = np.math.ceil(val_data_gen.samples / val_data_gen.batch_size)
predictions = model.predict_generator(val_data_gen, steps=test_steps_per_epoch)


# save our model
model.save("Models/" + model_nom + ".h5")

#Convert the model file to Tensorflow Lite 
if tflite :
    model1 = tf.keras.models.load_model("Models/" + model_nom + ".h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model1)
    tflite_model = converter.convert()
    open("Models/" + model_nom + ".tflite","wb").write(tflite_model)

# create a csv file to write report 
# (Timte, Model, Epochs, Batch Size, Number Classes, Avg Accuracy, Loss, Optimizer, Dropout, Learning Rate)
# This conditions to deal with the problem of 'acc' and 'accuracy' in different environements
with open('report.csv', 'a') as reportFile:
    if 'acc' in history.history:                     
        my_accuracy = history.history['acc'][0]
    if 'accuracy' in history.history:
        my_accuracy = history.history['accuracy'][0]

    if ('accuracy' in history.history) or ('acc' in history.history):
        x = datetime.datetime.now()
        info = [x, model_nom, epochs, batch_size, units2,
                my_accuracy, history.history['loss'][0], optimizer, dropout1, dropout2, learningrate]
        writer = csv.writer(reportFile, delimiter=',', lineterminator='\n')
        writer.writerow(info)
    else:
        print("Couldn't save the report!!")
    
    print(my_accuracy)  # Will help us in the K-fold cv script (K-fold.sh)