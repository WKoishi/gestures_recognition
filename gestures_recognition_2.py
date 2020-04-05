from cv2 import cv2
from time import clock
from math import acos,pi,log
import scfun
import numpy as np

#创建一个黑框，未使用
def Create_Outer_Mask(cap):
    _,Outer_mask=cap.read()
    Outer_mask=cv2.cvtColor(Outer_mask,cv2.COLOR_BGR2GRAY)
    f_size=Outer_mask.shape
    row=int(f_size[0]*0.05)
    col=int(f_size[1]*0.1)
    Outer_mask[:row,:]=0
    Outer_mask[-row:,:]=0
    Outer_mask[:,:col]=0
    Outer_mask[:,-col:]=0
    Outer_mask[row:-row,col:-col]=255
    return Outer_mask

#背景移除
def Remove_Background(model,frame):
    fgmask =model.apply(frame)
    kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
#    kernel_2 =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel_3 =cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    fgmask =cv2.dilate(fgmask,kernel_3,iterations=1)  #膨胀5*5
    fgmask =cv2.erode(fgmask,kernel,iterations=1)  #腐蚀3*3
#    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel_2)
#    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
#    fgmask=cv2.GaussianBlur(fgmask,(3,3),0)
    res =cv2.bitwise_and(frame,frame,mask=fgmask)
    return res

#识别肤色
def Bodyskin_Detect_Otsu(frame):
    ycrcb=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    cr=ycrcb[:,:,1]
    _,skin=cv2.threshold(cr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel_1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel_2=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    skin=cv2.morphologyEx(skin,cv2.MORPH_OPEN,kernel_2)
    skin=cv2.dilate(skin,kernel_1,iterations=1)

    return skin


def ConvexHull_Cal(contour):

    def IsTriangle(a,b,c):
        return a+b>c and a+c>b and b+c>a

    point_list=[]
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

    if len(point_list)>=2:
        for it_1,it_2 in zip(point_list,point_list[1:]+point_list[:1]):
            CA=scfun.Eucledian_Distance(it_1[1],it_1[2])  #far to end
            AB=scfun.Eucledian_Distance(it_1[2],it_2[1])  #end to next far
            
            if CA+AB>=radius:
                BC=scfun.Eucledian_Distance(it_1[1],it_2[1])  #far to far，为底边
                if IsTriangle(CA,AB,BC):
                    angle=acos((CA**2 + AB**2 - BC**2)/(2*CA*AB))

                    if angle<=pi/3:
                        ndefects+=1

    return ndefects


#寻找符合条件的轮廓
#按轮廓的面积从大到小，检索深度为3
def Find_Contour(img):
    ndefect_ls=[]
    MIN_AREA=3000  #检测的轮廓的最小面积

    contours,_ =cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #当len为0时表示没有找到轮廓，当len大于60时表示受到的干扰过大
    # #避免当contours为空时引发max函数错误而退出程序的情况
    length=len(contours)
    if 0<length<=60:
        contours.sort(key=lambda x: cv2.contourArea(x),reverse=True)
        
        if cv2.contourArea(contours[0])>=MIN_AREA:
            ndefect=ConvexHull_Cal(contours[0])
            ndefect_ls.append(ndefect)
            
            if length>=2 and cv2.contourArea(contours[1])>=MIN_AREA:
                ndefect=ConvexHull_Cal(contours[1])
                ndefect_ls.append(ndefect)

            if length>=3 and cv2.contourArea(contours[2])>=MIN_AREA:
                ndefect=ConvexHull_Cal(contours[2])
                ndefect_ls.append(ndefect)
            
            index_list=[ndefect_ls.index(x) for x in ndefect_ls if 0<x<=5]  #去除算出凸包大于5的结果

            if index_list:
                if len(index_list)>=2:
                    right_ls=[ndefect_ls[i] for i in index_list]
                    m_index=ndefect_ls.index(max(right_ls))
                else:
                    m_index=0
            
                return ndefect_ls[m_index],contours[m_index]

    return 0,0


#手势跟踪与识别
def Gestures_Detect(hand):

    ndefects,large_cout=Find_Contour(hand)
    if ndefects==0:
        ndefects=11  #返回contours为空的信息，只作调试用
        center=tuple([a//2 for a in reversed(hand.shape)])  #返回图像的中心坐标
        return hand,ndefects,center
    
    
    black2 = np.ones(hand.shape, np.uint8) #创建黑色幕布
    cv2.drawContours(black2,large_cout,-1,(255,255,255),2) #绘制白色轮廓
    cv2.imshow('large_cout',black2)
    
    '''
    test=scfun.Fourier_Descriptor(large_cout[:,0,:],Normalize=True)
    similar=scfun.Eucledian_Distance(test,fourier_des_ls[0])
    print('{:.5f}  {:.5f}'.format(similar,log(similar)))
    '''
    M=cv2.moments(large_cout)
    center=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))  #手部的质心坐标

    x,y,w,h = cv2.boundingRect(large_cout)

    hand=cv2.cvtColor(hand,cv2.COLOR_GRAY2BGR)  #将灰度图像转换为BGR以显示绿色方框
    hand = cv2.rectangle(hand,(x,y),(x+w,y+h),(0,255,0),2)

    return hand,ndefects,center

#识别结果滤波器，比如"2"只有在30次识别中出现15次以上识别结果才会被认定为2
def Ges_Num_Detect(new,buf,length):
    for i in range(length-1):
        buf[i]=buf[i+1]
    buf[length-1]=new
    for num in [1,2,3,4,5]:
        times=buf.count(num)
        if times>=int(length/2):
            return num
    return 13


def main():
    last_time=0
    R=70
    length=30
    ges_num_buf=[0]*length
    
    cap=cv2.VideoCapture(0)  #启动摄像头
    model=cv2.createBackgroundSubtractorKNN(history=300,detectShadows=False)  #背景移除KNN模型
    
    while(1):
        
        _,frame =cap.read()
        
        start=clock()
        #frame =cv2.bilateralFilter(frame,5,50,100)  #双边滤波
        xframe =Remove_Background(model,frame)
        hand =Bodyskin_Detect_Otsu(xframe)
        
        afhand,ges_num,hand_center=Gestures_Detect(hand)
        real_ges_num=Ges_Num_Detect(ges_num,ges_num_buf,length=length)

        frame_center=tuple([a//2 for a in reversed(hand.shape)])  #返回图像的中心坐标
        cv2.line(afhand,hand_center,frame_center,(255,0,0),3)
        cv2.circle(afhand,frame_center,R,(255,255,0),1)
        if scfun.Eucledian_Distance(hand_center,frame_center)>=R:
            angle=scfun.Azimuth_Angle(hand_center,origin=frame_center)
            cv2.putText(afhand,str(real_ges_num),(0,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        else:
            angle=0
        
        cv2.imshow('koishi',afhand)
        cv2.imshow('origin',xframe)
        
        finish=clock()
        this_time,last_time=scfun.Filter(finish-start,last_time,rate=0.2)
        print('{:.5f}  {}  {:.5f}  {}  {:.5f}'.format(ges_num, real_ges_num,
                                            finish-start, hand_center, angle))
        
        key=cv2.waitKey(1)&0XFF
        if key == 27:
            break  #按esc退出
        elif key == ord('q'):
            cv2.imwrite('hand_test.png',hand)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

