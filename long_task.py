
from scipy.signal import convolve2d
import os
import cv2
import glob
import time
import pickle
import shutil
import numpy as np
import traceback
import logging
import skimage.io as io
#from PIL import Image as PI
from wand.image import Image as WI
from wand.display import display
from pathlib import Path
from apputil import setup_logger
from segmenter.pre_processing import *
from imutils import resize as im_resize
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize, thin
from skimage.filters import threshold_otsu, gaussian, median, threshold_yen
import numpy as np
import math
from skimage.restoration import estimate_sigma
from time import sleep

#from wrapt_timeout_decorator import *
value=None

@timeout(1)
def long_task(message):
    import time as tim
    print(message)
    print("1")
    for i in range(1,10):
        print("2")
        tim.sleep(1)
        print('{} seconds have passed'.format(i))
        value = i+i
    return value
        
def estimate_noise2(img):
    #img = cv2.imread(image_path)
    return estimate_sigma(img, multichannel=True, average_sigmas=True)

def estimate_noise(I):
    H, W = I.shape
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
    return sigma


img = cv2.imread('sent_images/Wed_31_03_2021_224507.png', 0)
print('OpenCV: ', img.shape)
val = long_task("yo")
print(val)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_buffer = imgf = imgmat = None
segmented_staves=[]
img_m = np.array(img)
noise = estimate_noise(img_m)
print(f"Manual Sigma: {noise}")
noise = estimate_noise2(img_m)
print(f"Scikit Sigma: {noise}")
print(value)
# with WI.from_array(img) as im:
#     img_buffer = np.asarray(bytearray(im.make_blob("PNG")), dtype=np.uint8)
#     ret, mat = binarize_image(img_buffer)
#     with WI(blob=mat) as timg:
#         imgf = mat
#         timg.deskew(0.4*im.quantum_range)
#         imgf = np.array(timg)
#         img_buffer = np.asarray(bytearray(timg.make_blob("PNG")), dtype=np.uint8)
#         imgmat = cv2.imdecode(img_buffer, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Image',imgmat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# imgmat = get_thresholded(imgmat, threshold_otsu(imgmat))
