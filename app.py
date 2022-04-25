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
    output=np.argmax(preds, axis=1)
    if output==0:
        print("character is ञ")
    elif output==1:
        print("character is ट")
    elif output==2:
        print("character is ठ")
    elif output==3:
        print("character is ड")
    elif output==4:
        print("character is ढ")
    elif output==5:
        print("character is ण")
    elif output==6:
        print("character is त")
    elif output==7:
        print("character is थ")
    elif output==8:
        print("character is द")
    elif output==9:
        print("character is ध")
    elif output==10:
        print("character is क")
    elif output==11:
        print("character is न")
    elif output==12:
        print("character is प")
    elif output==13:
        print("character is फ")
    elif output==14:
        print("character is ब")
    elif output==15:
        print("character is भ")
    elif output==16:
        print("character is म")
    elif output==17:
        print("character is य")
    elif output==18:
        print("character is र")
    elif output==19:
        print("character is ल")
    elif output==20:
        print("character is व")
    elif output==21:
        print("character is ख")
    elif output==22:
        print("character is श")
    elif output==23:
        print("character is ष")
    elif output==24:
        print("character is स")
    elif output==25:
        print("character is ह")
    elif output==26:
        print("character is क्ष")
    elif output==27:
        print("character is त्र")
    elif output==28:
        print("character is ज्ञ")
    elif output==29:
        print("character is ग")
    elif output==30:
        print("character is घ")
    elif output==31:
        print("character is ङ")
    elif output==32:
        print("character is च")
    elif output==33:
        print("character is छ")
    elif output==34:
        print("character is ज")
    elif output==35:
        print("character is झ")
    elif output==36:
        print("character is ०")
    elif output==37:
        print("character is १")
    elif output==38:
        print("character is २")
    elif output==39:
        print("character is ३")  
    elif output==40:
        print("character is ४")
    elif output==41:
        print("character is ५")
    elif output==42:
        print("character is ६")
    elif output==43:
        print("character is ७")
    elif output==44:
        print("character is ८")
    elif output==45:
        print("character is ९")
    return output

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
