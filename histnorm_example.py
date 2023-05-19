# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:38:17 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from digitalimageprocesslib import histfunc

def histnorm(img, img_t):
    shape = img.shape
    cdf = histfunc.cumsum(img)/shape[0]/shape[1]
    cdf = (cdf*255).astype(np.uint8)
    img = cdf[img] 

    cdf_t = histfunc.cumsum(img_t)/shape[0]/shape[1]
    cdf_t = (cdf_t*255).astype(np.uint8)
    cdf_o = np.zeros(256)
    for k in range(256):
        cdf_o[cdf_t[k]] = k
    for k in range(1,256):
        if cdf_o[k] == 0:
            cdf_o[k] = cdf_o[k-1]
    img = cdf_o[img]
    return img, cdf_o

if __name__ == '__main__':
    img_target = cv2.imread("standard_test_images/jetplane.tif", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("standard_test_images/peppers_gray.tif", cv2.IMREAD_GRAYSCALE)
    img_out = histfunc.histnorm(img, img_target)
    
    hist = histfunc.calhist(img)
    hist_target =  histfunc.calhist(img_target)
    hist_out =  histfunc.calhist(img_out)
    # print(img_out.max())

    img_color = cv2.imread("standard_test_images/peppers_color.tif")
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    img_color_t = cv2.imread("standard_test_images/lena_color_512.tif")
    img_color_t = cv2.cvtColor(img_color_t, cv2.COLOR_BGR2RGB)
    img_color_out = histfunc.histnorm(img_color, img_color_t)

    hist_color = histfunc.calhist(img_color)
    hist_color_t =  histfunc.calhist(img_color_t)
    hist_color_out =  histfunc.calhist(img_color_out)

    #################### figure #################

    plt.figure(dpi=72,figsize=(8,6))
    plt.subplot(4,3,1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplot(4,3,2)
    plt.imshow(img_target, cmap='gray')
    plt.axis('off')
    plt.title('Target')
    plt.subplot(4,3,3)
    plt.imshow(img_out, cmap='gray')
    plt.axis('off')
    plt.title('Normalization')
    plt.subplot(4,3,7)
    plt.imshow(img_color)
    plt.axis('off')
    plt.subplot(4,3,8)
    plt.imshow(img_color_t)
    plt.axis('off')
    plt.subplot(4,3,9)
    plt.imshow(img_color_out)
    plt.axis('off')

    plt.subplot(4,3,4)
    plt.plot(hist,'black')
    plt.subplot(4,3,5)
    plt.plot(hist_target, 'black')
    plt.subplot(4,3,6)
    plt.plot(hist_out, 'black')

    cmap = ['r', 'g', 'b']
    plt.subplot(4,3,10)
    for k in range(3):
        plt.plot(hist_color[:,k], cmap[k])
    plt.subplot(4,3,11)
    for k in range(3):
        plt.plot(hist_color_t[:,k], cmap[k])
    plt.subplot(4,3,12)
    for k in range(3):
        plt.plot(hist_color_out[:,k], cmap[k])
    plt.show()