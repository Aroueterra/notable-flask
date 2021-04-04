#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import cv2
import math
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
from apputil import setup_logger, estimate_noise
from imutils import resize as im_resize
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize, thin
from skimage.filters import threshold_otsu, gaussian, median, threshold_yen
from .staff import calculate_thickness_spacing, remove_staff_lines, coordinator
from wrapt_timeout_decorator import *
from skimage.restoration import estimate_sigma

logging.basicConfig(filename='logs/slicer.log', level=logging.DEBUG)
log = setup_logger('segmenter', '/logs/segmenter.log')

#@timeout(20)
def deskew_and_binarize(img):
    import io
    import cv2
    import time as tim
    import numpy as np
    from PIL import Image
    from wand.image import Image as WI
    from segmenter.pre_processing import binarize_image
    img_buffer=None
    with WI.from_array(img) as im:
        im.deskew(0.4*im.quantum_range)
        pil_image = Image.open(io.BytesIO(im.make_blob("png")))                               
        np_image = np.array(pil_image)
        ret3, result = cv2.threshold(np_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return result
    
#@timeout(24)
def segment(imgmat):
    segmenter = Segmenter(imgmat)
    segments = segmenter.regions_with_staff
    return segments

def countpix(img):
    wht = cv2.countNonZero(img)
    height, width = img.shape
    px = height*width
    blk = px - wht
    return wht, blk

def Slice(cv_img):
    start_time = time.time()
    img_buffer = imgmat = None
    segmented_staves=[]
    try:
        log.info("SLICER: binarization start " + str(time.time() - start_time))
        imgmat = deskew_and_binarize(cv_img) 
        np_img = np.array(imgmat)
        noise = estimate_noise(np_img)
        wht, blk = countpix(result)
        log.info(f"SLICER: noise: {noise} ")
        log.info(f"SLICER: white: {wht} {blk} ")
        print(f"{wht} {blk} {noise}")
        if (blk > wht):
            log.error("ERROR: black pixel overflow")
            raise ValueError("black pixel overflow")
        log.info(f'File: noise level: {noise}')
        if(noise >= 6):
            raise ValueError("irregular noise levels")
    except Exception as e:
        log.error("ERROR: deskew and binarize error")
        log.error("".join(traceback.TracebackException.from_exception(e).format()))
        log.error(traceback.format_exc())
        raise
    imgmat = get_thresholded(imgmat, threshold_otsu(imgmat))
    try:
        log.info("SLICER: segmentation start " + str(time.time() - start_time)) 
        segments = segment(imgmat)
    except Exception as e:
        log.error("ERROR: segment error")
        log.error("".join(traceback.TracebackException.from_exception(e).format()))
        log.error(traceback.format_exc())
        raise
    log.info("SLICER: beginning cropping " + str(time.time() - start_time)) 
    try:
        for i, img in enumerate(segments):
            log.info("SLICER: cropped 1more image" + str(time.time() - start_time)) 
            segmented_staves.append(crop(img))
    except Exception as e:
        log.error("ERROR: crop and minify error ")
        log.error("".join(traceback.TracebackException.from_exception(e).format()))
        log.error(traceback.format_exc())
        raise
    log.info("SLICER: work completed " + str(time.time() - start_time)) 
    return segmented_staves

if __name__ == '__main__':
    Slice(r"C:\Users\aroue\Downloads\Documents\@ML\Sheet Music\goodsheet\pgws.png")





