from __future__ import division, print_function
import os, sys
import glob
import re
import numpy as np
import cv2

#load keras packages
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

#Flask packages
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model_path = "first_model.model"

model = load_model(model_path)
model._make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #input_img = image.load_img(img_path, target_size = (100,100))
    x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (100,100))
    x = x/255.0
    #Preprocess
    x = x.reshape(-1, 100, 100, 1)
    #Load to model
    preds = model.predict(x)

    return preds

@app.route('/', methods = ['GET'])
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        #get the file 
        f = request.files['file']
        # save the file to './uploads'
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #predict
        pred = model_predict(img_path=file_path, model=model)

        CATEGORIES = ['Cat', 'Dog']

        #pred_class = decode_predictions(pred, top=1)

        result = CATEGORIES[int(pred[0][0])]

        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)






