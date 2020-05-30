#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :otsu_threshold.py
@说明        :OTSU算法及肤色识别
@时间        :2020/04/19
@作者        :温震霆，陈炫华
@第三方库     : numpy 1.18.1; opencv-python 4.2.0
'''

from cv2 import cv2
import numpy as np
from dynamic_histogram import DynamicHist
from time import perf_counter
#import matplotlib.pyplot as plt

'''利用OTSU算法计算直方图中设定区间[sta:fin]的最佳阈值'''
def OTSU_Threshold(img,sta,fin):

    hist = cv2.calcHist([img],[0],None,[fin-sta],[sta,fin])
    hist=hist[:,0]
    total = np.sum(hist)
    current_max, threshold = 0, 0
    sumF, sumB = 0, 0
    sumT = np.inner(hist,np.arange(0,fin-sta))
    weightB, weightF = 0, 0
    varBetween, meanB, meanF = 0, 0, 0

    for i in range(0,fin-sta):
        weightB += hist[i]
        weightF = total - weightB
        if weightF == 0:
            break
        sumB += i*hist[i]
        sumF = sumT - sumB
        meanB = sumB/weightB
        meanF = sumF/weightF
        varBetween = weightB * weightF
        varBetween *= (meanB-meanF)*(meanB-meanF)
        if varBetween > current_max:
            current_max = varBetween
            threshold = i 
    threshold+= sta+1

    return threshold


'''肤色识别'''
def Bodyskin_Detect_Otsu(frame):
    ycrcb=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    cr=ycrcb[:,:,1]
    Y=ycrcb[:,:,0]

    sta=perf_counter()
    DynamicHist(cr,'dynamic hist',[127,168],removeInZero=False)
    fin=perf_counter()
    print(fin-sta)
    
    cr=cv2.GaussianBlur(cr,(7,7),0)
    thresh=OTSU_Threshold(cr,127,168)  #在给定的范围内计算阈值
    _,skin=cv2.threshold(cr,thresh,255,cv2.THRESH_BINARY)
    _,skin2=cv2.threshold(cr,168,255,cv2.THRESH_BINARY_INV)  #上限
    skin=cv2.bitwise_and(skin,skin2)
    
    kernel=np.ones((5,5),np.uint8)
    skin=cv2.morphologyEx(skin,cv2.MORPH_OPEN,kernel)
    skin=cv2.morphologyEx(skin,cv2.MORPH_CLOSE,kernel)
    
    return skin
