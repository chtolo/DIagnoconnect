from flask import Flask, render_template, jsonify, request, flash, redirect
import pickle
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
current_temperature = 21
app = Flask(__name__)

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('notebooks/diabetes.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('notebooks/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 15:
        model = pickle.load(open('notebooks/brainstroke.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
        

@app.route("/")
def home():
    return render_template('index.html')
@app.route("/about",methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route("/contact",methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabet.html')
@app.route("/stroke",methods=['GET', 'POST'])
def stroke():
    return render_template('stroke.html')

@app.route("/heart", methods=['GET', 'POST'])
def heart():
    return render_template('heart.html')


@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except:
        message = "Please enter valid Data"
        return render_template("error.html", message = message)

    return render_template('predict.html', pred = pred)
@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predictor", methods=['GET', 'POST'])
def predictor():
    return render_template('predictor.html')
        

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumonia():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img_size = 150
                model = load_model("notebooks/pneumo.h5")
                img = request.files['image']
                img_arr = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                resized_arr = resized_arr / 255
                resized_arr = resized_arr.reshape(-1, img_size, img_size, 1)

                # Make predictions using the model
                predictions = model.predict(resized_arr)
                pred = (0 if predictions[0]<0.5 else 1)
                
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pred_pneumonia.html', pred = pred)

if __name__ == '__main__':
	app.run(debug = True)