from generate_aug_data import *
import matplotlib.pylab as plt   
from im_operators import *
from visualize import *
from skimage import data, img_as_float
from skimage import exposure
from config import *
##### CONFIG ########
show_4 = False
showFalse=False
showDiff = False
showRaw = False
average=False
generator = generator_fast
UsePCAKmeans = True
PCAKmeansModel = "model_kmeans_pca_1.pkl" 
###### CONFIG #########3

if showRaw:
    crop = 8.0
    for i in xrange(1,10000):
        ex, label = load_img_det(i)
        if label[-1] == '0' or showFalse:
            print label
            print ex.shape
            im = [im_crop(j, crop) for j in ex.reshape(ImageChannels,64,64)]
            if showDiff:
                print im[1].shape
                print im[1]
                print im[1] - im[0]
                tmp = im[1] - im[0]
                print im[1]
                im[2] = im[1]
                im[1] = tmp
                im[3] = im[3] - im[0]

            if show_4:
                show_4_ex(im, label, title=label[-1])
            else:
                print "Showing raw 1"
                plt.subplot(3,1,1)
                plt.imshow(im[0])
                plt.title(label[-1])
                plt.colorbar()
                plt.subplot(3,1,2)
                plt.hist(im[0].flatten(), 256, range=(0,255)) 
                plt.subplot(3,1,3)
                plt.imshow(exposure.equalize_hist(im[0]))
                plt.show()
             
            #plt.title(str(label[-1]))
            #plt.imshow(im_crop(ex.reshape(4,64,64)[0],4.0), cmap='hot')
            #plt.show() 

else:
    N = 2100

    train_set_x, train_set_y, test_set_x, test_set_y = \
        get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator, add_x_extra=True)

    train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
    train_set_x = train_set_x[:, 0:train_set_x.shape[1]-ExtraColumns]
    test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
    test_set_x = test_set_x[:, 0:test_set_x.shape[1]-ExtraColumns]


    train_set_x_pca, test_set_x_pca = None, None
    pca, kmeans = None, None
    F = None
    if UsePCAKmeans: 
        ipixels = ImageChannels*ImageSideFinal*ImageSideFinal
        print "Loading PCA"
        pca, kmeans = cPickle.load(open(PCAKmeansModel, "r"))
        #TODO: add extra columns
        print "Transforming train"
        train_set_x_pca = kmeans.transform(pca.transform(train_set_x[:,0:ipixels]))
        print "Transforming test"
        test_set_x_pca = kmeans.transform(pca.transform(test_set_x[:,0:ipixels]))
        # Add pca variables
        F = pca.inverse_transform(kmeans.cluster_centers_)

    import matplotlib.pylab as plt
    for id, (ex, label) in enumerate(zip(train_set_x, train_set_y)):
            print ex

            ex = ex[0:ImageChannels*ImageSideFinal**2]
            im = [j for j in ex.reshape(ImageChannels, ImageSideFinal, ImageSideFinal)]
            if average: 
                avg = im[0]+im[1]+im[2]+im[3]
                im = [avg/4.0, avg/4.0, avg/4.0, avg/4.0]
            if label[-1] == 0 or showFalse:
                if show_4:
                    show_4_ex(im, None, title=str(label))
                else:
                    plt.subplot(3,1,1)
                    plt.imshow(im[0])
                    plt.title(label[-1])
                    plt.colorbar()
                    plt.subplot(3,1,2)
                    plt.hist(im[0].flatten(), 256, range=(0,255)) 
                    plt.subplot(3,1,3)
                    plt.imshow(exposure.equalize_hist(im[0]))
                    plt.show()
                
