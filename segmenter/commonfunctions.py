import os
import cv2
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from wand.color import Color
from PIL import Image, ImageChops
from wand.image import Image as WI
from matplotlib.pyplot import bar
from skimage.feature import canny
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.filters import threshold_otsu, gaussian, median
from skimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion, closing, opening, square, skeletonize, disk



def copy(src, dst):
    shutil.rmtree(dst + "\\melody")
    print(dst + " " + src)
    shutil.copytree(os.path.join(src, "output\\"), os.path.join(dst, "melody\\"))

def crop_image(input_image, output_image, start_x, start_y, width, height):
    """Pass input name image, output name image, x coordinate to start croping, y coordinate to start croping, width to crop, height to crop """
    input_img = Image.open(input_image)
    
    
    box = (start_x, start_y, start_x + width, start_y + height)
    cropped_img = input_img.crop(box)
    
    baseheight = 128
    hpercent = (baseheight / float(cropped_img.size[1]))
    wsize = int((float(cropped_img.size[0]) * float(hpercent)))
    resized_img = cropped_img.resize((wsize, baseheight), Image.ANTIALIAS)    
    resized_img.save(output_image +".png")
    
def save_slice(i,output_path,img):
    plt.rcParams["figure.figsize"] = (20,15)
    plt.gca().set_axis_off()
    plt.gca().set_title("")
    fig=plt.imshow(img,interpolation='nearest')
    plt.savefig(output_path,
    bbox_inches='tight', pad_inches=0, format='png', dpi=600)
    

def crop(path):
    with Image.open(path) as img:
        img_mat = np.asarray(img)
        with WI.from_array(img_mat) as im:
            im.trim(Color("WHITE"))
            im.save(filename=path)
    with Image.open(path) as cropped_img:
        cropped_img = Image.open(path)
        #print (f"cropped: {cropped_img.size[0]} x {cropped_img.size[1]}" + str(type(cropped_img)))
        baseheight = 155
        hpercent = (baseheight / float(cropped_img.size[1]))
        wsize = int((float(cropped_img.size[0]) * float(hpercent)))
        resized_img = cropped_img.resize((wsize, baseheight), Image.ANTIALIAS)    
        #print (f"resized: {resized_img.size[0]} x {resized_img.size[1]}")
        resized_img.save(path, quality=100)

def binarize_image(img):
    mat = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    ret3, bin_img = cv2.threshold(mat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return cv2.imencode(".jpg", bin_img)

def make_image(data, outputname, size=(128, 200), dpi=80):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.gray()
    ax.imshow(data, aspect='equal')
    plt.show()
    
def show_images(images, titles=None):
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.imsave('test.png', image, cmap = plt.cm.gray)
        #plt.savefig('a.pdf')
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    
def showHist(img):
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def gray_img(img):
    '''
    img: rgb image
    return: gray image, pixel values 0:255
    '''
    gray = rgb2gray(img)
    if len(img.shape) == 3:
        gray = gray*255
    return gray


def otsu(img):
    '''
    Otsu with gaussian
    img: gray image
    return: binary image, pixel values 0:1
    '''
    blur = gaussian(img)
    otsu_bin = 255*(blur > threshold_otsu(blur))
    return (otsu_bin/255).astype(np.int32)


def get_gray(img):
    gray = rgb2gray(np.copy(img))
    return gray


def get_thresholded(img, thresh):
    return 1*(img > thresh)
    


def histogram(img, thresh):
    hist = (np.ones(img.shape) - img).sum(dtype=np.int32, axis=1)
    _max = np.amax(hist)
    hist[hist[:] < _max * thresh] = 0
    return hist


def get_line_indices(hist):
    indices = []
    prev = 0
    for index, val in enumerate(hist):
        if val > 0 and prev <= 0:
            indices.append(index)
        prev = val
    return indices


def get_region_lines_indices(self, region):
    indices = get_line_indices(histogram(region, 0.8))
    lines = []
    for line_index in indices:
        line = []
        for k in range(self.thickness):
            line.append(line_index+k)
        lines.append(line)
    self.rows.append([np.average(x) for x in lines])
