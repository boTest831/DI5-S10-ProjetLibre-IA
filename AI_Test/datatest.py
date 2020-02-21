from PIL import Image
import numpy as np
import pandas as pd
#from scipy.misc import imsave  #can't work
import imageio
#img = imageio.imread(myImage)


#ignore warning
def silence_imageio_warning(*args, **kwargs):
    pass

def loadData():
    # load
    train_data = pd.read_csv('data/train.csv') #il y a des bruits sur image
    data0 = train_data.iloc[0, 1:-1]
    data1 = train_data.iloc[1, 1:-1]
    data0 = np.matrix(data0)
    data1 = np.matrix(data1)
    data0 = np.reshape(data0, (40, 40))
    data1 = np.reshape(data1, (40, 40))
    #imsave('test1.jpg', data0)
    #imsave('test1.jpg', data1)
    imageio.imsave('test0.jpg', data0)
    imageio.imsave('test1.jpg', data1)
def PictureToArray():
    # imgge to matrice
    image = Image.open('test0.jpg')
    matrix = np.asarray(image)
    print(matrix)
    # matrice to image
    # image1 = Image.fromarray(matrix)
    # image1.show()
if __name__ == '__main__':
    imageio.core.util._precision_warn = silence_imageio_warning
    loadData()
