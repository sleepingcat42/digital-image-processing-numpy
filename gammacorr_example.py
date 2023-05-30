# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:11:55 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from digitalimageprocesslib import histfunc

if __name__ == '__main__':
    img = cv2.imread("standard_test_images/peppers_gray.tif", cv2.IMREAD_GRAYSCALE)
    img1 = histfunc.gamma_corr(img, 0.5)
    img2 = histfunc.gamma_corr(img, 2)
    img3 = histfunc.gamma_corr(img1, 2)

     ###################### figure #####################
    plt.figure(figsize=(6,6))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.title('original image')

    plt.subplot(2, 2, 2)
    plt.imshow(img1, cmap = 'gray')
    plt.axis('off')
    plt.title('gamma = 0.5')


    plt.subplot(2, 2, 3)
    plt.imshow(img2, cmap = 'gray')
    plt.axis('off')
    plt.title('gamma = 2')

    plt.subplot(2, 2, 4)
    plt.imshow(img3, cmap = 'gray')
    plt.title('gamma correction ')
    plt.axis('off')
    plt.show()