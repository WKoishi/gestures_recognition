#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :phi_contour.py
@说明        :轮廓处理
@时间        :2020/04/19
@作者        :温震霆，陈炫华
@第三方库     : numpy 1.18.1; opencv-python 4.2.0
'''

from cv2 import cv2
import numpy as np
from math import acos,pi,sqrt

'''计算x,y两点的欧氏距离'''
def Eucledian_Distance(x,y):
    return sqrt(sum([(a-b)**2 for a,b in zip(x,y)]))


'''轮廓的凸包和凹陷处理函数'''
def ConvexHull_Cal(contour):

    IsTriangle = lambda a,b,c: a+b>c and a+c>b and b+c>a  #任意两边和必须大于第三边

    point_list=[]
    convex_angle_ls=[]  #凸包角度list
    concave_angle_ls=[]  #凹陷角度list
    ndefects=0

    epsilon = 0.003*cv2.arcLength(contour,True)
    contour = cv2.approxPolyDP(contour,epsilon,True)  #轮廓近似，Douglas-Peucker算法
    hull=cv2.convexHull(contour,returnPoints=False)
    defects=cv2.convexityDefects(contour,hull)
    _,radius=cv2.minEnclosingCircle(contour)
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,_ =defects[i,0]
            sta=tuple(contour[s][0])
            end=tuple(contour[e][0])
            far=tuple(contour[f][0])
            point_list.append([sta,far,end])

    #下面的角边标示含义见文件夹里的图片说明
    if len(point_list)>=2:
        for it_1,it_2 in zip(point_list,point_list[1:]+point_list[:1]):
            CA=Eucledian_Distance(it_1[1],it_1[2])  #far to end
            AB=Eucledian_Distance(it_1[2],it_2[1])  #end to next far
            #凸包角度
            if radius<=CA+AB<=2*radius:
                BC=Eucledian_Distance(it_1[1],it_2[1])  #far to 2nd far，为底边
                if IsTriangle(CA,AB,BC):
                    angle=acos((CA**2 + AB**2 - BC**2)/(2*CA*AB))
                    convex_angle_ls.append(angle)
            #凹陷角度
            DC=Eucledian_Distance(it_1[0],it_1[1])  #sta to far
            if radius<=DC+CA<=2*radius:
                DA=Eucledian_Distance(it_1[0],it_1[2])  #sta to end，为底边
                if IsTriangle(DC,CA,DA):
                    angle=acos((CA**2 + DC**2 - DA**2)/(2*CA*DC))
                    concave_angle_ls.append(angle)

        convex_angle=[x for x in convex_angle_ls if pi/18<=x<=pi/6]  #凸包角度:10度至30度
        convex_len=len(convex_angle)
        concave_angle=[x for x in concave_angle_ls if pi/18<=x<=pi/3.5]
        concave_len=len(concave_angle)

        if convex_len==1 and concave_len==0:
            ndefects=1
        elif 1<convex_len<=3 and 0<concave_len<=2:
            ndefects=concave_len+1
        elif convex_len>=3 and concave_len==3:
            mid_angle=[x for x in concave_angle_ls if pi/4 < x <= pi/2.5]  #寻找大拇指与食指间的角度
            if len(mid_angle)==1:
                ndefects=5
            elif len(mid_angle)==0:
                ndefects=4
        else:
            ndefects=0

    return ndefects


'''寻找符合条件的轮廓'''
#目前是通过处理凸包来区分轮廓
#按轮廓的面积从大到小，检索深度为3
def Find_Contour(img):
    ndefect_ls=[]
    MIN_AREA=2000  #检测的轮廓的最小面积
    depth=3

    contours,_ =cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #当len为0时表示没有找到轮廓，当len大于60时表示受到的干扰过大
    # #避免当contours为空时引发max函数错误而退出程序的情况
    length=len(contours)
    if 0<length<=60:
        contours.sort(key=lambda x: cv2.contourArea(x),reverse=True)

        for i in range(depth):
            if length>i and cv2.contourArea(contours[i])>=MIN_AREA:
                ndefect=ConvexHull_Cal(contours[i])
                ndefect_ls.append(ndefect)
            else:
                break
            
        if ndefect_ls:
            index_list=[ndefect_ls.index(x) for x in ndefect_ls if 0<x<=5]  #去除算出凸包数为0和大于5的结果

            if index_list:
                if len(index_list)>=2:
                    right_ls=[ndefect_ls[i] for i in index_list]
                    m_index=ndefect_ls.index(max(right_ls))
                elif len(index_list)==1:
                    m_index=index_list[0]

                return ndefect_ls[m_index],contours[m_index]

    return 0,0


#识别结果滤波器，采用列表递推滤波的形式
#比如"2"只有在30次识别中出现15次以上识别结果才会被认定为2
def Ges_Num_Filter(new,buf,length):
    for i in range(length-1):
        buf[i]=buf[i+1]
    buf[length-1]=new
    for num in [1,2,3,4,5]:
        times=buf.count(num)
        if times>=int(length/2):
            return num
    return 13

