import os
import tensorflow as tf
import numpy as np
from skimage.util import img_as_float
import FuzzyCMeans as fm
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, redirect, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
main_page = "home.html"

@app.route("/")
def home():
    return render_template(main_page)

@app.route("/app", methods=['GET', 'POST'])
def application():
    if request.method == 'GET':
        return render_template("index.html")
    
    return render_template(main_page)

@app.route("/about", methods=['GET', 'POST'])
def about():
    if request.method == 'GET':
        return render_template("about.html")
    
    return render_template(main_page)

model = load_model('TrainedFile/TrainFile.h5')

def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor was found"
	elif classNo==1:
		return "You have a Brain Tumor"

def getResult(img):
    image = cv2.imread(img)

    """
    image = img_as_float(image)
    x = np.reshape(image, (image.shape[0] * image.shape[1], 3), order='F')
    cluster_n = 6
    expo = 6
    min_err = 0.001
    max_iter = 500
    verbose = 0

    m, c = fm.fcm(x, cluster_n, expo, min_err, max_iter, verbose)
    m = np.reshape(m, (image.shape[0], image.shape[1]), order='F')

    simg = fm.calc_median(image,m,verbose)
    image_c = ((simg + 1) * 255 / 2).astype('uint8')
    """
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image= np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict_classes(input_img)
    return result

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 
                                'uploads', 
                                secure_filename(f.filename)
                                )

        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None

if __name__ == "__main__":
    app.run()