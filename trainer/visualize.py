import matplotlib.pylab as pl
from matplotlib import animation
import numpy as np
from skimage import exposure
from im_operators import *
import os
from data_api import *
import json


def load_img_det(i):
    raw_values = [float(x) for x in open("data/{0}_img.raw".format(i)).
        read().split(" ") if len(x) > 0]
    det = None
    with open("data/{0}.det".format(i)) as f:
        det = f.read().split(" ")
    return np.array(raw_values).reshape(4, 64, 64), det

def get_img_det(i):
    raw_values = [float(x) for x in open("data/{0}_img.raw".format(i)).
        read().split(" ") if len(x) > 0]
    im1_raw = np.array(raw_values).reshape(4, 64, 64)
    det = None
    with open("data/{0}.det".format(i)) as f:
        det = f.read().split(" ")

    im1_raw = (im1_raw / max(raw_values))

    im1 = [0,0,0,0]

    for i in xrange(4):
        im1[i] = im1_raw[i]
        im1[i] = ndimage.gaussian_filter(im1_raw[i], sigma=1.1)
        im1[i] = im_crop(im1[i], (64/ (2*max(2,float(det[-6])))))

    return im1 , det

def show_4_ex(im1, det=None, title=None):
    f, ((p1,p2),(p3,p4)) = pl.subplots(2,2)
    im_show = p1.imshow(im1[0], cmap='hot')
    im_show = p2.imshow(im1[1], cmap='hot')
    im_show = p3.imshow(im1[2], cmap='hot')
    im_show = p4.imshow(im1[3], cmap='hot')
    if det: pl.title("Detection "+" result "+det[-1]+ " pixel width "+det[-6])
    if title: pl.title(title)
    pl.show()



def show_4(i, im2 = None):
    det = None
    im1 = None
    if im2 is None:

        im1, det = get_img_det(i)
        print "Detection "+str(i)+" result "+det[-1]
    else:
        im1 = im2

    f, ((p1,p2),(p3,p4)) = pl.subplots(2,2)

    if im2 is None:
        pl.title("Detection "+str(i)+" result "+det[-1]+ " pixel width "+det[-6])

    im_show = p1.imshow(im1[0], cmap='hot')
    im_show = p2.imshow(im1[1], cmap='hot')
    im_show = p3.imshow(im1[2], cmap='hot')
    im_show = p4.imshow(im1[3], cmap='hot')

    pl.show()


def show(i):
    im1, det = get_img_det(i)
    im_show = pl.imshow(im1[0], cmap='hot')
    pl.title("Detection "+str(i)+" result "+det[-1]+ " pixel width "+det[-6])
    pl.show()

def show_anim(i):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = pl.figure()

    im1, det = get_img_det(i)

    print det[-1]


    im_show = pl.imshow(im1[0], cmap='hot')
    # initialization function: plot the background of each frame
    # initialization function: plot the background of each frame


    def init():
        im_show.set_data(im1[0])
        return [im_show]

    # animation function.  This is called sequentially
    def animate(i):
        a=im_show.get_array()
        a[:,:] = im1[i%4]
        pl.title("Frame "+str(i%4)+" detection "+det[-1] + " pixel width "+det[-6])
        im_show.set_array(a)
        return [im_show]


    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=10, interval=1000)

    #for i in cycle(xrange(4)):
    #    animate(i)
    #    time.sleep(1)
    pl.show()







def calc_imbalance():
    detc, ndetc = 0, 0
    for i in xrange(11331):
        print i
        img, det = get_img_det(i+1)
        if det[-1]=="1": detc += 1
        else: ndetc += 1
    print detc, " ", ndetc




#im, det = get_img_det(20)
##
##print im
##
#show_4(20, generator_crop_flip_8fold(im[1]))
#show_4(20)
##show_anim(1000)
