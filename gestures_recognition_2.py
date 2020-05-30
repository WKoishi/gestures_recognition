from cv2 import cv2
from time import perf_counter
from math import acos,pi,log
import scfun
import numpy as np
import matplotlib.pyplot as plt

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


def Bodyskin_Detect(frame,CrCbHist):

    frame=cv2.blur(frame,(5,5))
    ycrcb=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    cr=ycrcb[:,:,1]
    cb=ycrcb[:,:,2]

    result=CrCbHist[cr,cb]  #《这是神句》

    kernel_1=np.ones((3,3),np.uint8)
    kernel_2=np.ones((5,5),np.uint8)
    result= cv2.erode(result,kernel_1,iterations=1)
    result= cv2.dilate(result,kernel_2,iterations=1)
    #result=cv2.medianBlur(result,3)
    
    return result


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
            CA=scfun.Eucledian_Distance(it_1[1],it_1[2])  #far to end
            AB=scfun.Eucledian_Distance(it_1[2],it_2[1])  #end to next far
            #凸包角度
            if CA+AB>=radius:
                BC=scfun.Eucledian_Distance(it_1[1],it_2[1])  #far to 2nd far，为底边
                if IsTriangle(CA,AB,BC):
                    angle=acos((CA**2 + AB**2 - BC**2)/(2*CA*AB))
                    convex_angle_ls.append(angle)
            #凹陷角度
            DC=scfun.Eucledian_Distance(it_1[0],it_1[1])  #sta to far
            if DC+CA>=radius:
                DA=scfun.Eucledian_Distance(it_1[0],it_1[2])  #sta to end，为底边
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


#寻找符合条件的轮廓
#按轮廓的面积从大到小，检索深度为3
def Find_Contour(img):
    ndefect_ls=[]
    MIN_AREA=2000  #检测的轮廓的最小面积

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
                elif len(index_list)==1:
                    m_index=index_list[0]

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
    
    capture=cv2.VideoCapture(1)  #启动摄像头
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    
    skinCrCbHist=np.zeros((256,256),dtype=np.uint8)
    cv2.ellipse(skinCrCbHist,(113,156),(25,16),43,0,360,255,-1)  #肤色CrCb椭圆模型
    
    while(1):
        
        _,frame =capture.read()

        start=perf_counter()
        #frame =cv2.bilateralFilter(frame,5,50,100)  #双边滤波
        #xframe =Remove_Background(model,frame)
        
        sta=perf_counter()
        #hand =Bodyskin_Detect_Otsu(frame)
        hand=Bodyskin_Detect(frame,skinCrCbHist)
        fin=perf_counter()
        
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
        cv2.imshow('origin',frame)
        
        finish=perf_counter()
        this_time,last_time=scfun.Filter(finish-start,last_time,rate=0.2)
        print('{:.5f}  {}  {:.5f}  {}  {:.5f}'.format(ges_num, real_ges_num,
                                            finish-start, hand_center, angle))
        #print(fin-sta)
        
        key=cv2.waitKey(1)&0XFF
        if key == 27:
            break  #按esc退出
        elif key == ord('q'):
            cv2.imwrite('hand_test.png',hand)
    
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

