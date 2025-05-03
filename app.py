import os
import tensorflow as tf
import numpy as np
from keras import preprocessing
from PIL import Image
import cv2
from keras import models
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model = models.load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img):
    image = cv2.imread(img)
    image = cv2.resize(image, (64, 64)) 
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    
    # Normalize the image
    input_img = input_img / 255.0
    
    # Predict the class
    predictions = model.predict(input_img)
    print("Predictions:", predictions)  # Print prediction scores for debugging
    
    # Get the class with the highest probability
    result = np.argmax(predictions, axis=1)
    
    return result[0] 


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)

        # Get prediction result
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
