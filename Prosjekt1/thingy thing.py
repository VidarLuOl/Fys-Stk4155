import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

terrain1 = imread('SRTM_data_Norway_1.tif')
terrain2 = imread('SRTM_data_Norway_2.tif')


def dataTerrain(img):
    x = np.linspace(0,1,np.shape(img)[1])
    y = np.linspace(0,1,np.shape(img)[0])
    x,y = np.meshgrid(x,y)
    return x,y,img


x,y,z = dataTerrain(terrain1)

plt.contour(x,y,z)