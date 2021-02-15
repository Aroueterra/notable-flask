#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import cv2
import glob
import time
import pickle
import numpy as np
from .box import Box
#from .fit import predict
from .connected_componentes import  *
from .pre_processing import *
from .commonfunctions import *

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

def Slice(cv_img):
    start_time = time.time()
    img_buffer=None
    imgf=None
    imgmat=None
    segmented_staves=[]
    print("===============================BINARIZATION==============================")
    with Image.from_array(cv_img) as im:
        img_buffer = np.asarray(bytearray(im.make_blob("JPEG")), dtype=np.uint8)
        ret, mat = binarize_image(img_buffer)
        with Image(blob=mat) as timg:
            imgf = mat
            #timg.save(filename="otsu.jpg")
            timg.deskew(0.4*im.quantum_range)
            #timg.save(filename="otsu2.jpg")
            imgf = np.array(timg)
            img_buffer = np.asarray(bytearray(timg.make_blob("JPEG")), dtype=np.uint8)
            imgmat = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)

    print("==================================SLICE==================================")
    imgmat = get_thresholded(imgmat, 245)
    segmenter = Segmenter(imgmat)
    imgs_with_staff = segmenter.regions_with_staff
    show_images([imgs_with_staff[0]])
    mypath = Path().absolute()
    file_path = str(mypath) + '\\segmenter\\output\\'
    delete_path = str(mypath) + '\\segmenter\\output'
    absolute_path = Path(file_path)
    print("Output of slices: " + file_path)
    remove_dir = os.listdir(delete_path)
    for item in remove_dir:
        if item.endswith(".png"):
            os.remove(os.path.join(delete_path, item))
    print("==================================CROP===================================")
    for i, img in enumerate(imgs_with_staff):
        plt.rcParams["figure.figsize"] = (20,15)
        plt.gca().set_axis_off()
        plt.gca().set_title("")
        fig=plt.imshow(imgs_with_staff[i],interpolation='nearest')
        output_path = file_path+'slice'+str(i)+'.png'
        plt.savefig(output_path,
        bbox_inches='tight', pad_inches=0, format='png', dpi=600)
        print("    ++Image generated in " + str(time.time() - start_time))
        print(output_path)
        crop(output_path)
        segmented_staves.append(Path(output_path))

    print("PROCESS COMPLETED in: " + str(time.time() - start_time))
    return segmented_staves

if __name__ == '__main__':
    Slice(r"C:\Users\aroue\Downloads\Documents\@ML\Sheet Music\goodsheet\pgws.png")





