import os
import cv2
import time
import config
import traceback
import ctc_utils
import numpy as np
import zipfile
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image
from ml_model import ML
from PIL import ImageFont
from PIL import ImageDraw
import silence_tensorflow.auto
from melody import generateWAV
from segmenter.slicer import Slice
from flask_ngrok import run_with_ngrok
from flask import Flask,request,send_from_directory,render_template,flash,redirect,url_for,send_file,jsonify
from apputil import normalize, resize, sparse_tensor_to_strs, elements, allowed_file, compress

# GLOBAL ACCESS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
# SETUP APPLICATION
UPLOAD_FOLDER = 'input'

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "key"


run_with_ngrok(app)
model = ML(config.model,config.voc_file,config.input_dir,
           config.slice_dir,config.classification,config.seq)
session = model.setup()


# HOME
@app.route("/")
def root():
    return render_template('index.html')
# IMAGE REQUEST
@app.route('/img/<filename>')
def send_img(filename):
    return send_from_directory('', filename)
# ANDROID REQUEST
@app.route('/android/predict', methods = ['GET', 'POST'])
def login():
    return 'Yeah, it works.'

# POST TEST
@app.route('/test', methods = ['GET', 'POST'])
def test_output():
    app.logger.info('TEST: headers')
    app.logger.info(request.headers)
    app.logger.info('TEST: success')
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                "archive.zip", as_attachment=True)
    return jsonify(success=1,error="none",error_type="")


# MODEL PREDICTION
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        count = list(request.files)
        app.logger.info('POST: success')
        app.logger.info('# of files in request: ' + str(len(count)))
        app.logger.info(str(request.headers))
        if 'file' not in request.files:
            app.logger.error('No file part in request.files')
            return "No file part in request.files"
        app.logger.info('File: exists')
        f = request.files['file']
        if f.filename == '':
            return "No selected file"
        else:
            app.logger.info("File: name = " + f.filename)
        if f and allowed_file(f.filename):
            app.logger.info('File: exists | Format: matching')
            start_time = time.time()
            try:
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
                img = Image.open(request.files['file'].stream).convert('RGB')
            except Exception:
                app.logger.error("ERROR: image input error")
                app.logger.error("".join(traceback.TracebackException.from_exception(e).format()))
                app.logger.error(traceback.format_exc())
                return "Image input error"
            try:
                np_img = np.array(img)
                print(np_img.shape)
#                 gry_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
                save = np.array(cv_img)
                cv2.imwrite('test0.jpg', save)
                #gry_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                app.logger.error("ERROR: conversion error")
                app.logger.error("".join(traceback.TracebackException.from_exception(e).format()))
                app.logger.error(traceback.format_exc())
                return "Conversion error"
            try:
                all_predictions = model.predict(cv_img)
            except Exception as e:
                app.logger.error('ERROR: prediction exception' + str(e))
                app.logger.error("".join(traceback.TracebackException.from_exception(e).format()))
                return "Prediction error"
            try:
                generateWAV(all_predictions, "false")
                memory_file = compress('data/melody')
            except Exception as e:
                app.logger.error('ERROR: audio exception'  + str(e))
                app.logger.error("".join(traceback.TracebackException.from_exception(e).format()))
                return "Audio/compression error"
            app.logger.info("All processes completed: " + str(time.time() - start_time) )
            #return 'Prediction: success'
            return send_file(memory_file,
                    attachment_filename='archive.zip',
                    as_attachment=True)
        else:
            app.logger.error('File: incompatible format')
    else:
        app.logger.error('POST: failed, found get request')
    return 'Exiting the program, server did literally nothing'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__=="__main__":
    handler = RotatingFileHandler('logs/app.log', maxBytes=10000, backupCount=1) 
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.DEBUG)
    app.run()
