# Import packages
from keras.preprocessing.image import img_to_array
from numpy import *
import argparse
import os
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import validators
import requests
from io import BytesIO
import time

def prediction(url):
            # L is list that contains all logos names as a string lower case  
    L = [os.listdir("Dataset/train")[i].lower()
        for i in range(len(os.listdir("Dataset/train/")))]
    class_names = sorted(L, reverse=False)

    # Affecting each args to a variable and loading the model choosing before
    model_nom = "sansAug2"

    #Download an image from an given URL or locally from a directory an resize it
    if validators.url(url) :
        with requests.get(url) as url:
            img = image.load_img(BytesIO(url.content), target_size=(400, 400))
    else :
        img = image.load_img(url, target_size=(400, 400))
    img = np.expand_dims(img, axis=0)

    # t1 = time.time()
    # predicting the giving image either with tflite or h5 format
    interpreter = tf.lite.Interpreter(model_path="./" + model_nom + ".tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    prediction = prediction[0]
    print(prediction)

    
    # Looping for all brands
    L1=[]
    for i in range(len(prediction)):
        L1 += [[str(class_names[i]), float(prediction[i])]]

    return L1
        # print(str(url) + "," + str(class_names[i]) + "," + str(prediction[i]))
    # t2 = time.time()
    # print("The prediction time : {}s".format(t2-t1))

#prediction("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/DHL-Fahrzeug.jpg/1200px-DHL-Fahrzeug.jpg")
    
