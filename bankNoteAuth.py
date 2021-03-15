# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)#first step where flask starts

pickle_in = open(r'C:\Users\navan\bkntauth.pkl','rb')
model = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome all"

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = model.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is " + str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv("https://raw.githubusercontent.com/krishnaik06/Dockers/master/TestFile.csv")
    prediction = model.predict(df_test)
    return "The predicted value for csv is " + str(list(prediction))

if __name__ == '__main__':
    app.run()