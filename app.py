#from flask import Flask,request,send_from_directory,render_template
#from flask_ngrok import run_with_ngrok
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import shutil
import uvicorn
import numpy as np
import model_utility
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from pyngrok import ngrok
from fastapi import FastAPI
from fastapi.logger import logger
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class Settings(BaseSettings):
    # ... The rest of our FastAPI settings
    BASE_URL = "http://localhost:8000"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"

def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass
        
settings = Settings()


# Initialize the FastAPI app for a simple web server
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#run_with_ngrok(app)


voc_file = "vocabulary_semantic.txt"
model = "semantic_model/semantic_model.meta"

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Read the dictionary
dict_file = open(voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

# Restore weights
saver = tf.train.import_meta_graph(model)
saver.restore(sess,model[:-5])

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)



@app.get('/')
async def root():
    return {'hello': 'world'}

@app.post('/predict')
async def predict():
    f = request.files['file']
    img = f
    image = Image.open(img).convert('L')
    image = np.array(image)
    image = resize(image, HEIGHT)
    image = normalize(image)
    image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)
    seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
    prediction = sess.run(decoded,
                      feed_dict={
                          input: image,
                          seq_len: seq_lengths,
                          rnn_keep_prob: 1.0,
                      })
    str_predictions = sparse_tensor_to_strs(prediction)
    array_of_notes = []

    for w in str_predictions[0]:
        array_of_notes.append(int2word[w])
    notes=[]
    for i in array_of_notes:
        if i[0:5]=="note-":
            if not i[6].isdigit():
                notes.append(i[5:7])
            else:
                notes.append(i[5])
    img = Image.open(img).convert('L')
    size = (img.size[0], int(img.size[1]*1.5))
    layer = Image.new('RGB', size, (255,255,255))
    layer.paste(img, box=None)
    img_arr = np.array(layer)
    height = int(img_arr.shape[0])
    width = int(img_arr.shape[1])
    print(img_arr.shape[0])
    draw = ImageDraw.Draw(layer)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("Aaargh.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    j = width / 9
    for i in notes:
        draw.text((j, height-40), i, (0,0,0), font=font)
        j+= (width / (len(notes) + 4))
    layer.save("annotated.png")
    return templates.TemplateResponse("result.html")

if __name__=="__main__":
    #ngrok_tunnel = ngrok.connect(8000)
    
    ngrok.set_auth_token("1nM8RPS113N7wSZvyYJ3VpgADFl_53evML7vJRGGYJA8PhCK6")
    public_url = ngrok.connect('8000').public_url
    print('Public URL:', public_url)
    logger.info("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, '8000'))
    # Update any base URLs or webhooks to use the public ngrok URL
    settings.BASE_URL = public_url
    init_webhooks(public_url)
    #nest_asyncio.apply()
    #host="localhost", 
    uvicorn.run(
        "app:app", 
        port=8000,
        reload=False    
    )
