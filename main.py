#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :main.py
@说明        :简单的手势识别，适合的摄像头输出帧的大小：640*480
@时间        :2020/04/19
@作者        :温震霆，陈炫华
@第三方库     : numpy 1.18.1; opencv-python 4.2.0
'''

from phi_contour import Find_Contour,Eucledian_Distance,Ges_Num_Filter
from otsu_threshold import Bodyskin_Detect_Otsu
from cv2 import cv2
import numpy as np
from time import perf_counter


length=30  #滤波长度
ges_num_buf=[0]*length  #滤波器的缓存列表
err_times=0
time_add,nor_times=0,0

capture=cv2.VideoCapture(0)  #启动摄像头

while(1):

    ret,frame=capture.read()

    if ret==False:
        err_times+=1
        print("无法获取帧")
        if err_times>=5:
            break
        continue

    start=perf_counter()  #开始计时
    
    skin=Bodyskin_Detect_Otsu(frame)

    ndefects,right_cont=Find_Contour(skin)

    if ndefects==0:
        ndefects=11  #返回contours为空的信息，只作调试用
        center=tuple([a//2 for a in reversed(skin.shape)])  #返回图像的中心坐标
    else:
        '''
        black2 = np.ones(skin.shape, np.uint8) #创建黑色幕布
        cv2.drawContours(black2,right_cont,-1,(255,255,255),2) #绘制白色轮廓
        cv2.imshow('right_cont',black2)
        '''
        M=cv2.moments(right_cont)
        center=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))  #手部的质心坐标

        x,y,w,h = cv2.boundingRect(right_cont)
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    #经滤波器过滤后的可信度较高的手势数字表示
    real_ges_num = Ges_Num_Filter(ndefects,ges_num_buf,length=length)

    finish=perf_counter()  #结束计时
    time_add+=finish-start
    nor_times+=1

    cv2.imshow('skin',skin)
    cv2.imshow('origin',frame)

    #print('第{}次帧处理: {:.5f}  {}  Time:{:.6f}'.format(nor_times,ndefects, real_ges_num,finish-start))

    if nor_times>=1000:
        time_avg=time_add/1000
        print('处理一帧的平均用时(s): {}'.format(time_avg))
        print('\n')
        break

    #按esc退出
    key=cv2.waitKey(1)&0XFF
    if key == 27:
        break  
    elif key == ord('q'):
        pass

capture.release()
cv2.destroyAllWindows()


