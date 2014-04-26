from generate_aug_data import *
import matplotlib.pylab as plt   
from im_operators import * 
from visualize import *
for i in xrange(1,10000):
    ex, label = load_img_det(i)
    if label[-1] == '0':
        print label
        print ex.shape
        im = [im_crop(j, 3.0) for j in ex.reshape(4,64,64)]
        show_4_ex(im, label, title=label[-1])
        #plt.title(str(label[-1]))
        #plt.imshow(im_crop(ex.reshape(4,64,64)[0],4.0), cmap='hot')
        #plt.show() 


