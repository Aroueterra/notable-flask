import os
import cv2
import time
import config
import traceback
import ctc_utils
import numpy as np
import zipfile
import logging
from PIL import Image
from ml_model import ML
from PIL import ImageFont
from PIL import ImageDraw
from waitress import serve
import silence_tensorflow.auto
from melody import generate_WAV
from segmenter.slicer import Slice
from flask_ngrok import run_with_ngrok
from logging.handlers import RotatingFileHandler
from flask import Flask,request,send_from_directory,render_template,flash,redirect,url_for,send_file,jsonify
from apputil import normalize, resize, sparse_tensor_to_strs, elements, allowed_file, compress

# GLOBAL ACCESS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
# SETUP APPLICATION
UPLOAD_FOLDER = 'sent_images'
#static_url_path=''
heroku_app = None
try:
    heroku_app = Flask(__name__)
    logging.info('Starting up..')
    heroku_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    heroku_app.secret_key = "key"
except Exception as e:
    heroku_app.logger.exception(e)

#run_with_ngrok(heroku_app)
model = ML(config.model,config.voc_file,config.input_dir,
           config.slice_dir,config.classification,config.seq)
session = model.setup()


# HOME
@heroku_app.route("/")
def root():
    return render_template('index.html')
# IMAGE REQUEST
@heroku_app.route('/img/<filename>')
def send_img(filename):
    return send_from_directory('', filename)
# ANDROID REQUEST
@heroku_app.route('/android/predict', methods = ['GET', 'POST'])
def login():
    return 'Yeah, it works.'

# POST TEST
@heroku_app.route('/test', methods = ['GET', 'POST'])
def test_output():
    heroku_app.logger.info('TEST: headers')
    heroku_app.logger.info(request.headers)
    heroku_app.logger.info('TEST: success')
    return jsonify(success=1,error="none",error_type="")


# MODEL PREDICTION
@heroku_app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        count = list(request.files)
        heroku_app.logger.info('POST: success')
        heroku_app.logger.info('# of files in request: ' + str(len(count)))
        heroku_app.logger.info(str(request.headers))
        if 'file' not in request.files:
            heroku_app.logger.error('No file part in request.files')
            return "No file part in request.files"
        heroku_app.logger.info('File: exists')
        f = request.files['file']
        if f.filename == '':
            return "No selected file"
        else:
            heroku_app.logger.info("File: name = " + f.filename)
        if f and allowed_file(f.filename):
            heroku_app.logger.info('File: exists | Format: matching')
            start_time = time.time()
            try:
                f.save(os.path.join(heroku_app.config['UPLOAD_FOLDER'], f.filename))
                img = Image.open(request.files['file'].stream).convert('RGB')
            except Exception:
                heroku_app.logger.error("ERROR: image input error")
                heroku_app.logger.error("".join(traceback.TracebackException.from_exception(e).format()))
                heroku_app.logger.error(traceback.format_exc())
                return "Image input error"
            try:
                np_img = np.array(img)
                cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            except Exception as e:
                heroku_app.logger.error("ERROR: conversion error")
                heroku_app.logger.error("".join(traceback.TracebackException.from_exception(e).format()))
                heroku_app.logger.error(traceback.format_exc())
                return "Conversion error"
            try:
                all_predictions, segmented_staves = model.predict(cv_img)
            except Exception as e:
                heroku_app.logger.error('ERROR: prediction exception' + str(e))
                heroku_app.logger.error("".join(traceback.TracebackException.from_exception(e).format()))
                return "Prediction error"
            try:
                text_files, fullsong_file, song_files = generate_WAV(all_predictions, "false")
                memory_file = compress(fullsong_file, text_files, song_files, segmented_staves)
            except Exception as e:
                heroku_app.logger.error('ERROR: audio exception'  + str(e))
                heroku_app.logger.error("".join(traceback.TracebackException.from_exception(e).format()))
                return "Audio/compression error"
            heroku_app.logger.info("All processes completed: " + str(time.time() - start_time) )
            return send_file(memory_file,
                    attachment_filename='archive.zip',
                    as_attachment=True)
        else:
            heroku_app.logger.error('File: incompatible format')
    else:
        heroku_app.logger.error('POST: failed, found get request')
    return 'Exiting the program, server did literally nothing'


@heroku_app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(heroku_app.config['UPLOAD_FOLDER'],
                               filename)

if __name__=="__main__":
    port = int(os.environ.get('PORT', 33507))
    handler = RotatingFileHandler('logs/all_errors.log', maxBytes=10000, backupCount=1) 
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    heroku_app.logger.addHandler(handler)
    heroku_app.logger.setLevel(logging.DEBUG)
    print(f'serve {str(port)}')
    #print('serve')web: waitress-serve --port=$PORT app:app
    #heroku_app.run(debug=False, port=get_port(), host='0.0.0.0')host='0.0.0.0'
    serve(heroku_app, port=port)
    #print('serving')
#     heroku_app.run()
