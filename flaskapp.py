import json
import boto3
import numpy as np
import os, io
import PIL.Image as Image
import flask
from flask import Flask
from flask import request
from matplotlib.image import imsave
import sys, os

import matplotlib.pyplot as plt 

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from inferencing import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODEL='model_quant_12_03.tflite'
BUCKET="testlambdatf"
s3 = boto3.resource('s3')

app = Flask(__name__)


LABELS = ['BG', 'hair', 'nails']

#s3://imagemaskdump/mask/
@app.route('/predict/<username>/<img_name>')
def get_predictions(username=None, img_name=None):
    #username = request.args['username']
    #img_url = request.args['img_url']
#    image_url = request.args.get("img_url")
    print(img_name, username)
    image_url = username + '/'+img_name
    # image_url = "updohairstyle_143.jpeg"
    bucket = s3.Bucket(BUCKET)
    object = bucket.Object(image_url)
    response = object.get()

    img = Image.open(response['Body'])   
  
    # img = img[np.newaxis, ...]

    with tf.device("/cpu:0"):
        
        interpreter = tf.lite.Interpreter(model_path='model_quant_12_03.tflite')
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        img = img.resize((input_shape[1], input_shape[2]))
        input_data = np.expand_dims(img, axis=0)
        if len(input_data.shape) < 4:
            return flask.jsonify('Invalid Image Dimension. No. of image dimensions should be 3')
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]
        print(labels)
        print(top_k)
        print(results)
        result = {}
        for i in range(0, len(top_k)):
            print(labels[top_k[i]], float(results[top_k[i]] / 255.0))

        result["Model Output"] = {"class": labels[top_k[0]], "Confidence": float(results[top_k[0]] / 255.0)}
        return flask.jsonify(result)
