from cv2 import cv2
import numpy as np
import scfun
from math import acos,pi


def ConvexHull_Cal(contour):

    IsTriangle = lambda a,b,c: a+b>c and a+c>b and b+c>a  #任意两边和必须大于第三边

    point_list=[]
    convex_angle_ls=[]
    concave_angle_ls=[]

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
            CA=scfun.Eucledian_Distance(it_1[1],it_1[2])  #far to end
            AB=scfun.Eucledian_Distance(it_1[2],it_2[1])  #end to next far
            #凸包的角度
            if radius<=CA+AB<2*radius:
                BC=scfun.Eucledian_Distance(it_1[1],it_2[1])  #far to 2nd far，为底边
                if IsTriangle(CA,AB,BC):
                    angle=acos((CA**2 + AB**2 - BC**2)/(2*CA*AB))
                    convex_angle_ls.append(angle)
            #凹陷的角度
            DC=scfun.Eucledian_Distance(it_1[0],it_1[1])  #sta to far
            if radius<=DC+CA<2*radius:
                DA=scfun.Eucledian_Distance(it_1[0],it_1[2])  #sta to end，为底边
                if IsTriangle(DC,CA,DA):
                    angle=acos((CA**2 + DC**2 - DA**2)/(2*CA*DC))
                    concave_angle_ls.append(angle)

        convex_angle=[x for x in convex_angle_ls if pi/18<=x<=pi/6]  #凸包角度:10度至30度
        convex_len=len(convex_angle)
        concave_angle=[x for x in concave_angle_ls if pi/18<=x<=pi/3.5]
        concave_len=len(concave_angle)

        result=[convex_len,concave_len]
    
    else:
        result=[0,0]

    return result


#按轮廓的面积从大到小，检索深度为3
def Find_Contour(img_list):
    depth=3
    convexHull=[]
    Hu_list=[]
    contour_ls=[]
    MIN_AREA=2000  #检测的轮廓的最小面积

    for img in img_list:

        contours,_ =cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #当len为0时表示没有找到轮廓，当len大于60时表示受到的干扰过大
        # #避免当contours为空时引发max函数错误而退出程序的情况
        length=len(contours)
        if 0<length<=60:
            contours.sort(key=lambda x: cv2.contourArea(x),reverse=True)

            for i in range(depth):
                if length>i and cv2.contourArea(contours[i])>=MIN_AREA:
                    contour_ls.append(contours[i])
                    convexHull.append(ConvexHull_Cal(contours[i]))
                    M=cv2.moments(contours[i])
                    Hu_list.append(cv2.HuMoments(M))
                else:
                    break
    
    if Hu_list:
        Hu_array=np.array(Hu_list)
        Hu_array=np.squeeze(Hu_array)
    
    return contour_ls,convexHull,Hu_array

