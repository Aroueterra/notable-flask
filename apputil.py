import os
import cv2
import time
import shutil
import zipfile
import numpy as np
from io import BytesIO

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

def compress(file):
    file_path = file
    memory_file = BytesIO()
    rootdir = os.path.basename(file_path)
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
          for root, dirs, files in os.walk(file_path):
                    for file in files:
                        # Write the file named filename to the archive,
                        # giving it the archive name 'arcname'.
                        filepath   = os.path.join(root, file)
                        parentpath = os.path.relpath(filepath, file_path)
                        #arcname    = os.path.join(rootdir, parentpath)

                        zipf.write(filepath, file)
                        #zipf.write(os.path.join(root, file))
                              #zipf.write(os.path.join(root, file))
                            
    memory_file.seek(0)
    return memory_file
    