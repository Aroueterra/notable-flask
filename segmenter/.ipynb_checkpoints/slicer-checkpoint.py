#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import cv2
import glob
import time
import pickle
import shutil
import numpy as np
from .box import Box
#from .fit import predict
from .connected_componentes import  *
from .pre_processing import *
from .commonfunctions import *
import traceback
import logging
import skimage.io as io
from PIL import Image
from wand.image import Image
from .segmenter import Segmenter
from wand.display import display
from pathlib import Path
from imutils import resize as im_resize
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize, thin
from skimage.filters import threshold_otsu, gaussian, median, threshold_yen
from .staff import calculate_thickness_spacing, remove_staff_lines, coordinator
logging.basicConfig(filename='logs/slicer.log', level=logging.DEBUG)
def Slice(cv_img):
    start_time = time.time()
    img_buffer=None
    imgf=None
    imgmat=None
    segmented_staves=[]
    logging.info("SLICER: beginning binarization " + str(time.time() - start_time))
    try:
        with Image.from_array(cv_img) as im:
            img_buffer = np.asarray(bytearray(im.make_blob("JPEG")), dtype=np.uint8)
            ret, mat = binarize_image(img_buffer)
            with Image(blob=mat) as timg:
                imgf = mat
                timg.deskew(0.4*im.quantum_range)
                imgf = np.array(timg)
                img_buffer = np.asarray(bytearray(timg.make_blob("JPEG")), dtype=np.uint8)
                imgmat = cv2.imdecode(img_buffer, cv2.IMREAD_GRAYSCALE)
    except cv2.error as e:
        logging.error(traceback.format_exc())   
        logging.error("CV: read error")
        return
    cv2.imwrite('test2.jpg', imgmat)
    logging.info("SLICER: beginning segmentation " + str(time.time() - start_time)) 
    imgmat = get_thresholded(imgmat, threshold_otsu(imgmat))
    segmenter = Segmenter(imgmat)
    imgs_with_staff = segmenter.regions_with_staff
    show_images([imgs_with_staff[0]])
    mypath = Path().absolute()
    file_path = str(mypath) + '\\segmenter\\output\\'
    zip_path = str(mypath) + '\\data\\melody\\'
    zip_path1 = str(mypath) + '\\data\\melody'
    zip_path2 = str(mypath) + '\\data\\melody'
    delete_path = str(mypath) + '\\segmenter\\output'
    absolute_path = Path(file_path)
    
    remove_dir = os.listdir(delete_path)
    for item in remove_dir:
        if item.endswith(".png") or item.endswith(".jpg") or item.endswith(".jpeg"):
            os.remove(os.path.join(delete_path, item))
    remove_dir = os.listdir(zip_path1)
    for item in remove_dir:
        if item.endswith(".png") or item.endswith(".jpg") or item.endswith(".jpeg"):
            os.remove(os.path.join(zip_path1, item))
    logging.info("SLICER: beginning cropping " + str(time.time() - start_time)) 
    for i, img in enumerate(imgs_with_staff):
        output_path = file_path+'slice'+str(i)+'.png'
        zipped_path = zip_path+'slice'+str(i)+'.png'
        save_slice(i, output_path, img)
        
        save_slice(i, zipped_path, img)
        logging.info("SLICER: image++ in outputs " + str(time.time() - start_time)) 
        crop(output_path)
        crop(zipped_path)
        segmented_staves.append(Path(output_path))
    logging.info("SLICER: work completed " + str(time.time() - start_time)) 
    return segmented_staves

if __name__ == '__main__':
    Slice(r"C:\Users\aroue\Downloads\Documents\@ML\Sheet Music\goodsheet\pgws.png")





