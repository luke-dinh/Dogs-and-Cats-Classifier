from __future__ import division, print_function
import os, sys
import glob
import re
import numpy as np

#load keras packages
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

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

def prepare_path(img):
    img_size = 100
    img_array = image.img_to_array(img)
    new_array = img_array/255.0
    return new_array.reshape(-1,100,100,1)

def model_predict(img_path, model):
    input_img = prepare_path(img_path)

    #Preprocess
    x = np.expand_dims(input_img, axis=0)

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
        preds = model_predict(file_path, model)

        pred_class = decode_predictions(preds, top=1)

        result = str(pred_class[0][0][1])

        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)






