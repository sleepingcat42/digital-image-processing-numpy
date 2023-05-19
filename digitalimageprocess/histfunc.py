# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:18:55 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""
import numpy as np

def calhist_gray(img):
        
    M, N = img.shape
    hist = np.zeros(256)     
    # print(hist)
    for i in range(M):
        for j in range(N):
            hist[img[i, j]] += 1 
    return hist

def calhist(img):
    '''
    Parameters
    ----------
    img : np.array uint8
        Image

    Returns
    -------
    hist : np.array
        Histogram
    '''
    shape = img.shape
    # print(shape)

    if len(shape) == 3:
        hist = np.zeros([256, shape[2]])
        for k in range(shape[2]):
            hist[:,k] = calhist_gray(img[:,:,k])
        return hist
    elif len(shape) == 2:
        return calhist_gray(img)
            
def cumsum(img):
    histogram = calhist(img)
    cdf = np.zeros(256)
    cdf[0] = histogram[0]
    M, N = img.shape
    for k in range(1, 256):
        cdf[k] = cdf[k-1]+histogram[k]
    # cdf = cdf/M/N 
    return cdf


def histeq(img):
    shape = img.shape
    img_out = np.zeros_like(img)
    if len(shape) == 2:
        cdf = cumsum(img)/shape[0]/shape[1]
        cdf = (cdf*255).astype(np.uint8)
        img_out = cdf[img] 

    elif len(shape) == 3:
        # print('????')
        cdf = np.zeros([256, shape[2]])
        for k in range(shape[2]):
            cdf = cumsum(img[:,:,k])/shape[0]/shape[1]
            
            cdf = (cdf*255).astype(np.uint8)
            img_out[:,:, k] = cdf[img[:,:,k]]  

    return img_out


def histnorm_gray(img, img_t):
    shape = img.shape
    cdf = cumsum(img)/shape[0]/shape[1]
    cdf = (cdf*255).astype(np.uint8)
    img = cdf[img] 

    cdf_t = cumsum(img_t)/shape[0]/shape[1]
    cdf_t = (cdf_t*255).astype(np.uint8)
    cdf_o = np.zeros(256)
    for k in range(256):
        cdf_o[cdf_t[k]] = k
    for k in range(1,256):
        if cdf_o[k] == 0:
            cdf_o[k] = cdf_o[k-1]
    img = cdf_o[img]
    return img


def histnorm(img, img_t):
    shape = img.shape
    img_out = np.zeros_like(img)
    if len(shape) == 2:
        img_out = histnorm_gray(img, img_t)

    elif len(shape) == 3:
        for k in range(shape[2]):
            img_out[:,:, k] = histnorm_gray(img[:,:,k], img_t[:,:,k]) 
    return img_out.astype(np.uint8)