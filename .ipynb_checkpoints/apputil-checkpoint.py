import os
import cv2
import time
import shutil
import logging
import zipfile
import numpy as np
from io import BytesIO
from skimage.restoration import estimate_sigma

def format_error(e):
    e = str(e)
    infolen = (75) if len(e) > 75 else len(e)
    info = (e[:infolen] + '..')
    return info

def estimate_noise(img):
    return estimate_sigma(img, multichannel=True, average_sigmas=True)

def setup_logger(logger_name, log_file, level=logging.ERROR):
    l = logging.getLogger(logger_name)
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    log_file = basepath + log_file
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    #streamHandler = logging.StreamHandler()
    #streamHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    #l.addHandler(streamHandler)   
    return logging.getLogger(logger_name)
    
def sparse_tensor_to_strs(sparse_tensor):
    indices= sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]
    strs = [ [] for i in range(dense_shape[0]) ]
    string = []
    ptr = 0
    b = 0
    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]
        string.append(values[ptr])
        ptr = ptr + 1
    strs[b] = string
    return strs

def normalize(image):
    return (255. - image)/255.

def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img
def elements(array):
    return array.ndim and array.size

def allowed_file(filename):
    ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compress1(output_filename, dir_name):
    shutil.make_archive(output_filename, 'zip', dir_name)
    
def compress2(file_name):
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        files = result['files']
        for individualFile in files:
            data = zipfile.ZipInfo(individualFile['fileName'])
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, individualFile['fileData'])
    memory_file.seek(0)
    return send_file(memory_file, attachment_filename='capsule.zip', as_attachment=True)

def compress(fullsong_file, text_files, song_files, segmented_staves):
    #rootdir = os.path.basename(file_path)
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        txt_ctr = song_ctr = img_ctr = 0
        #Full_song
        zipf.writestr('full_song.wav', fullsong_file[0].getvalue())
        #Predictions
        for txt in text_files:
            file_name = f'predictions{str(txt_ctr)}.txt'
            zipf.writestr(file_name, BytesIO(txt).getvalue())
            txt_ctr+=1
        #Staff_song
        for song in song_files:
            file_name = f'staff{str(song_ctr)}.wav'
            zipf.writestr(file_name, song.getvalue())
            song_ctr+=1
        #Staff_images
        for img in segmented_staves:
            img_byte = BytesIO()
            img.save(img_byte, format='PNG')
            file_name = f'slice{str(img_ctr)}.png'
            zipf.writestr(file_name, img_byte.getvalue())
            img_ctr+=1
    memory_file.seek(0)
    return memory_file
    
    
