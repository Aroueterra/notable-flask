
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian, median
from skimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion, closing, opening, square, skeletonize, disk
from skimage.feature import canny
from skimage.transform import resize
from PIL import Image, ImageChops

def crop(path):
    #img = Image.fromarray(np.uint8(img))
    img = Image.open(path)
    #print(type(img))
    pixels = img.load()
    print (f"original: {img.size[0]} x {img.size[1]}")
    xlist = []
    ylist = []
    for y in range(0, img.size[1]):
        for x in range(0, img.size[0]):
            if pixels[x, y] != (255, 255, 255, 255):
                xlist.append(x)
                ylist.append(y)
    left = min(xlist)
    right = max(xlist)
    top = min(ylist)
    bottom = max(ylist)
    img = img.crop((left-10, top-10, right+10, bottom+10))
    print (f"cropped: {img.size[0]} x {img.size[1]}")
    img.save(path)
    #img = np.asarray(img)


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
    
    #plt.savefig(r'C:\Users\aroue\Downloads\Documents\@ML\Mozart\newout\foo.png', bbox_inches='tight')
    #plt.show()
    


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
