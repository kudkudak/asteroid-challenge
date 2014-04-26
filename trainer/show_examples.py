from generate_aug_data import *
import matplotlib.pylab as plt   
from im_operators import * 
from visualize import *
showFalse=True
showDiff = True
showRaw = False
average=False
generator = generator_fast

if showRaw:
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

else:
    N = 2000

    train_set_x, train_set_y, test_set_x, test_set_y = \
        get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator, add_x_extra=True)

    train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
    train_set_x = train_set_x[:, 0:train_set_x.shape[1]-ExtraColumns]
    test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
    test_set_x = test_set_x[:, 0:test_set_x.shape[1]-ExtraColumns]



    for ex, label in zip(train_set_x, train_set_y):
            im = [j for j in ex.reshape(ImageChannels, ImageSideFinal, ImageSideFinal)]
            if average: 
                avg = im[0]+im[1]+im[2]+im[3]
                im = [avg/4.0, avg/4.0, avg/4.0, avg/4.0]
            if label == 0 or showFalse:
                show_4_ex(im, None, title=str(label))
