#!/home/digimind/anaconda3/bin/python3.7
import tensorflow as tf
tf.get_logger().setLevel('ERROR');
from keras.preprocessing import image
import numpy as np
import os
import argparse
from numpy import *
import os.path
import shutil
import csv
import matplotlib.pyplot as plt
import time


#Add a cliparser to our bp.py 
#The modele_noun to choose a model from Models folder 
#And the path1 to get the path of the test folder 
parser = argparse.ArgumentParser(prog= 'test', description='This will help you to test your model')
parser.add_argument('modele_noun', type=str, help='Choose a model from Models folder')
parser.add_argument('path1',type=str, help='The path of the test folder')
parser.add_argument('--tflite', action='store_true', help='If you want to use tf lite extension')
args = parser.parse_args()

# L is list that contains all logos names as a string lower case  
L = [os.listdir("Dataset/train/")[i].lower() for i in range(len(os.listdir("Dataset/train/")))]
# Sorting the L list in ascending order 
class_names = sorted(L, reverse=False)

# Affecting each args to a variable and loading the model choosing before
model_nom, path1 = args.modele_noun, args.path1
if args.tflite :
        interpreter = tf.lite.Interpreter(model_path="Models/" + model_nom + ".tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
else :
        model = tf.keras.models.load_model("Models/" + model_nom + ".h5")
t1 = time.time()
# Looping for all images in that folder and predict each image
for x in os.listdir(args.path1):
        path2 = args.path1 + "/" + x
        img = image.load_img(path2, target_size=(150, 150))
        img = np.expand_dims(img, axis=0)
        if args.tflite :
                input_data = np.array(img, dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                predictions = predictions[0]
                prediction = max(predictions)
                indexs = predictions.tolist().index(prediction)
        else :
                indexs = model.predict_classes(img)[0];
                prediction = model.predict_proba(img)[0][indexs]
                       
        print(str(path2) + "," + str(class_names[indexs]) + "," + str(prediction))
t2 = time.time()
print("The prediction time is : {}s".format(t2-t1))




















