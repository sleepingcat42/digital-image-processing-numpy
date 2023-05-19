# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:33:06 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from digitalimageprocesslib import histfunc

if __name__ == '__main__':
    # grayscale image
    img = cv2.imread("standard_test_images/woman_darkhair.tif", cv2.IMREAD_GRAYSCALE)
    histogram = histfunc.calhist(img)
    cdf = histfunc.cumsum(img)
    
    img_norm = histfunc.histeq(img)
    histogram_norm = histfunc.calhist(img_norm)
    cdf_norm = histfunc.cumsum(img_norm)

    # color image
    img_color = cv2.imread("standard_test_images/lena_color_512.tif")
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    histogram_color = histfunc.calhist(img_color)
    cdf_color = np.zeros([256, 3])
    for k in range(3):
        cdf_color[:,k] = histfunc.cumsum(img_color[:,:,k])
        
    img_color_norm = histfunc.histeq(img_color)
    histogram_color_norm = histfunc.calhist(img_color_norm)
    cdf_color_norm = np.zeros([256, 3])
    for k in range(3):
        cdf_color_norm[:,k] = histfunc.cumsum(img_color_norm[:,:,k])

    ###################### figure #####################
    plt.figure(figsize=(16,8),dpi=72)
    plt.subplot(4,3,1)
    plt.imshow(img, cmap='gray')
    plt.ylabel('Original')
    plt.title('Image')

    plt.subplot(4,3,2)
    plt.plot(histogram, 'black')
    plt.ylabel('pixels')
    plt.xlabel('Grayscale')
    plt.xlim([0, 256])
    plt.title('HIstogram')

    plt.subplot(4,3,3)
    plt.plot(cdf, 'black')
    # plt.ylabel('pixels')
    plt.xlabel('Grayscale')
    plt.xlim([0, 256])
    plt.title('CDF')
    
    plt.subplot(4,3,4)
    plt.imshow(img_norm, cmap='gray')
    plt.ylabel('Normalization')
    
    plt.subplot(4,3,5)
    plt.plot(histogram_norm, 'black')
    plt.ylabel('pixels')
    plt.xlabel('Grayscale')
    plt.xlim([0, 256])

    plt.subplot(4,3,6)
    plt.plot(cdf_norm, 'black')
    # plt.ylabel('pixels')
    plt.xlabel('Grayscale')
    plt.xlim([0, 256])
    
    plt.subplot(4,3,7)
    plt.imshow(img_color, cmap='gray')
    plt.ylabel('Original')

    plt.subplot(4,3,10)
    plt.imshow(img_color_norm, cmap='gray')
    plt.ylabel('Normalization')

    plt.subplot(4,3,8)
    plt.plot(histogram_color[:,0], 'r')
    plt.plot(histogram_color[:,1], 'g')
    plt.plot(histogram_color[:,2], 'b')
    plt.ylabel('pixels')
    
    plt.subplot(4,3,9)
    plt.plot(cdf_color[:,0], 'r')
    plt.plot(cdf_color[:,1], 'g')
    plt.plot(cdf_color[:,2], 'b')
    # plt.ylabel('pixels')
    
    plt.subplot(4,3,11)
    plt.plot(histogram_color_norm[:,0], 'r')
    plt.plot(histogram_color_norm[:,1], 'g')
    plt.plot(histogram_color_norm[:,2], 'b')
    plt.ylabel('pixels')
    
    plt.subplot(4,3,12)
    plt.plot(cdf_color_norm[:,0], 'r')
    plt.plot(cdf_color_norm[:,1], 'g')
    plt.plot(cdf_color_norm[:,2], 'b')

    plt.show()

