import os
import cv2
import time
import config
import ctc_utils
import numpy as np
import zipfile
from PIL import Image
from ml_model import ML
from PIL import ImageFont
from PIL import ImageDraw
import silence_tensorflow.auto
from melody import generateWAV
from segmenter.slicer import Slice
from flask_ngrok import run_with_ngrok
from flask import Flask,request,send_from_directory,render_template,flash,redirect,url_for,send_file
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
# GET
@app.route('/users/<var>')
def hello_user(var):
    """
    this serves as a demo purpose
    :param user:
    :return: str
    """
    return "Wow, the GET works %s!" % var

# POST
@app.route('/api/post_some_data', methods=['POST'])
def get_text_prediction():
    """
    predicts requested text whether it is ham or spam
    :return: json
    """
    json = request.get_json()
    print(json)
    if len(json['text']) == 0:
        return jsonify({'error': 'invalid input'})

    return jsonify({'This is the KEY': json['This is the value?']})

# MODEL PREDICTION
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        count = list(request.files)
        print(len(count))
        #print(str(image_num))
        flash('POST SUCCESS')
        print("POST SUCCESS")
        print("PRINTING HEADERS")
        print(request.headers)
        print("PRINTING VALUES")
        for v in request.values:
            print(v)
        if 'file_name' not in request.files:
            print("FILE_NAME DOES NOT EXIST")
        else:
            fn = request.files['file_name']
            print(fn.filename)
            print(str(fn))
            
        if 'file' not in request.files:
            flash('FILE DOES NOT EXIST')
            print("FILE DOES NOT EXIST")
            return "No file part in request", 400
            #return redirect(request.url)
        print("FILE EXISTS REQUEST")
        f = request.files['file']
        
        print("READING FILE")
        
        print(f.filename)
        if f.filename == '':
            flash('No selected file')
            print("No selected file")
            return "No selected file", 400
            #return redirect(request.url)
        if f and allowed_file(f.filename):
            print("PREDICTING FILE")
            start_time = time.time()
            count = list(request.files)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
            img = Image.open(request.files['file'].stream).convert('RGB')
            np_img = np.array(img)
            cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            all_predictions = model.predict(cv_img)
            generateWAV(all_predictions, "false")
            memory_file = compress('data/melody')
            print("ALL PREDICTIONS COMPLETED in: " + str(time.time() - start_time) )    
            return send_file(memory_file,
                     attachment_filename='archive.zip',
                     as_attachment=True)
        else:
            print("FORMAT FAILURE")
    else:
        print("POST failed")
    return 'EXIT'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
# @app.route('/predict', methods = ['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         #READ INPUT
#         if 'file' not in request.files:
#             flash('No files')
#             return 'Maybe send a file?'
#             #return redirect(request.url)
#         f = request.files['file']
#         if f:
#             img = Image.open(request.files['file'].stream).convert('RGB')
#             np_img = np.array(img)
#             cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
#             all_predictions = model.predict(cv_img)
#             generateWAV(all_predictions, "false")
#         return render_template('result.html')
#     return render_template('result.html')

# x = "some data you want to return"
# return x, 200, {'Content-Type': 'text/css; charset=utf-8'}

# from flask import Response
# r = Response(response="TEST OK", status=200, mimetype="application/xml")
# r.headers["Content-Type"] = "text/xml; charset=utf-8"
# return r

if __name__=="__main__":
    app.run()
