# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:28:13 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""

import cv2
# import numpy as np
import matplotlib.pyplot as plt
from digitalimageprocesslib import histfunc


if __name__ == '__main__':
    img = cv2.imread("standard_test_images/peppers_gray.tif", cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    histogram = histfunc.calhist(img)
    
    img_color = cv2.imread("standard_test_images/peppers_color.tif")
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    histogram_color = histfunc.calhist(img_color)
    

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap = 'gray')
    
    plt.subplot(2, 2, 2)
    plt.plot(histogram, 'black')
    plt.ylabel('pixels')
    plt.xlabel('Grayscale')
    plt.xlim([0, 256])
    # plt.title('Histogram of grayscale image')
        
    plt.subplot(2,2,3)
    plt.imshow(img_color)
    
    plt.subplot(2,2, 4)
    plt.plot(histogram_color[:,0], 'r')
    plt.plot(histogram_color[:,1], 'g')
    plt.plot(histogram_color[:,2], 'b')
    plt.ylabel('pixels')
    # plt.xlabel('Grayscale')
    plt.xlim([0, 256])
    plt.show()
    
    # plt.title('Histogram of color image')