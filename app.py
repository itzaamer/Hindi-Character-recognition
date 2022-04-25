import sys
import os
import glob
import re
import numpy as np
import cv2
import keras
# Keras
from keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
Model_Path ='HindiModel1.h5'

# Load your trained model
model = tf.keras.models.load_model(Model_Path)



def model_predict(img_path, model):
    image = cv2.imread(img_path,0)/255
    image = np.asarray(image)
    img = cv2.resize(image,(32,32))
    img = np.asarray(img)
    # Preprocessing the image
    x = np.expand_dims(img, axis=0)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)

    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
