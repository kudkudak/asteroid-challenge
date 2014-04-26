from generate_aug_data import *
import matplotlib.pylab as plt   
from im_operators import * 
from visualize import *
showFalse=True
showDiff = True
crop = 8.0
for i in xrange(1,10000):
    ex, label = load_img_det(i)
    if label[-1] == '0' or showFalse:
        print label
        print ex.shape
        im = [im_crop(j, crop) for j in ex.reshape(4,64,64)]
        if showDiff:
            print im[1].shape
            print im[1]
            print im[1] - im[0]
            tmp = im[1] - im[0]
            print im[1]
            im[2] = im[1]
            im[1] = tmp
            im[3] = im[3] - im[0]
        show_4_ex(im, label, title=label[-1])
        #plt.title(str(label[-1]))
        #plt.imshow(im_crop(ex.reshape(4,64,64)[0],4.0), cmap='hot')
        #plt.show() 


