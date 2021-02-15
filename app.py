import os
import cv2
import config
import ctc_utils
import numpy as np
from PIL import Image
from ml_model import ML
from PIL import ImageFont
from PIL import ImageDraw
import silence_tensorflow.auto
from melody import generateWAV
from segmenter.slicer import Slice
from flask_ngrok import run_with_ngrok
from flask import Flask,request,send_from_directory,render_template,flash
from apputil import normalize, resize, sparse_tensor_to_strs, elements

# GLOBAL ACCESS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
# SETUP APPLICATION
app = Flask(__name__, static_url_path='')
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


#UPLOAD_FOLDER = 'static/upload'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#@app.route('/upload', methods=['GET','POST'])
#def upload():
#    if flask.request.method == "POST":
#        files = flask.request.files.getlist("file")
#        for file in files:
#            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

# MODEL PREDICTION
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        #READ INPUT
        if 'file' not in request.files:
            flash('No files')
            return redirect(request.url)
        f = request.files['file']
        if f:
            img = Image.open(request.files['file'].stream).convert('RGB')
            np_img = np.array(img)
            cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            all_predictions = model.predict(cv_img)
            generateWAV(all_predictions, "false")
        return render_template('result.html')

if __name__=="__main__":
    app.run()
