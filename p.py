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


#add a cliparser to our p.py (model_noun and the path)
parser = argparse.ArgumentParser(
    prog='test', description='This will help you to test your model')
parser.add_argument('modele_noun', type=str,
                    help='Chose a model from Models folder')
parser.add_argument('path', type=str, help='The path of the test image')
parser.add_argument('--tflite', action='store_true', help='If you want to use tf lite extension')

args = parser.parse_args()

# L is list that contains all logos names as a string lower case  
L = [os.listdir("Dataset/train/")[i].lower()
     for i in range(len(os.listdir("Dataset/train/")))]
class_names = sorted(L, reverse=False)

# Affecting each args to a variable and loading the model choosing before
model_nom, path = args.modele_noun, args.path

#Download an image from an given URL or locally from a directory an resize it
if validators.url(path) :
    with requests.get(path) as url:
        img = image.load_img(BytesIO(url.content), target_size=(150, 150))
else :
    img = image.load_img(path, target_size=(150, 150))
img = np.expand_dims(img, axis=0)

t1 = time.time()
# predicting the giving image either with tflite or h5 format
if args.tflite :
    interpreter = tf.lite.Interpreter(model_path="Models/" + model_nom + ".tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    prediction = prediction[0]
else :
    model = tf.keras.models.load_model("Models/" + model_nom + ".h5")
    prediction = model.predict_proba(img)[0]

# Looping for all brands
for i in range(len(prediction)):
    print(str(args.path) + "," + str(class_names[i]) + "," + str(prediction[i]))
t2 = time.time()
print("The prediction time : {}s".format(t2-t1))