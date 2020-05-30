from cv2 import cv2
from time import clock
from math import acos,pi,log
import scfun
import numpy as np

#目前没啥用的函数
def Sobel_Filter(img):
        sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        return sobel_xy


#生成手部样本的轮廓的描述
def Create_Hand_Sample_Contour():

    #加载手部素材图
    name_list=['hand_one.png','hand_two.png','hand_three.png','hand_four.png','hand_five.png']
    hand_img_ls=[]
    for name in name_list:
        hand_img_ls.append(cv2.imread('images/'+name,0))
    
    #生成手部轮廓
    contour_ls=[]
    fourdestor_ls=[]
    for hd in hand_img_ls:
        cout,_=cv2.findContours(hd,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        max_cout=max(cout,key=lambda x: cv2.contourArea(x))
        contour_ls.append(max_cout)

        descriptor=scfun.Fourier_Descriptor(max_cout[:,0,:],Normalize=True)
        fourdestor_ls.append(descriptor)
        
        black2 = np.ones((480,640), np.uint8) #创建黑色幕布
        cv2.drawContours(black2,max_cout,-1,(255,255,255),3) #绘制白色轮廓
        cv2.imshow('black2',hand_img_ls[1])
        cv2.waitKey(0)
        
    return contour_ls,fourdestor_ls

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

#寻找符合条件的轮廓
#按轮廓的面积从大到小，检索深度为2
def Find_Contour(img,sample_list,fourier_des_ls):

    def Similarity_cal(contour):
        descriptor=scfun.Fourier_Descriptor(contour[:,0,:],Normalize=True)
        sim_cout=[cv2.matchShapes(contour,sample,cv2.CONTOURS_MATCH_I1,0) for sample in sample_list]
        sim_four=[scfun.Eucledian_Distance(descriptor,des) for des in fourier_des_ls]
        return sim_cout,sim_four

    sign=False
    sign_2=False
    sim_cout_0=[]
    sim_four_0=[]
    sim_cout_1=[]
    sim_four_1=[]
    
    contours,_ =cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #当len为0时表示没有找到轮廓，当len大于60时表示受到的干扰过大
    # #避免当contours为空时引发max函数错误而退出程序的情况
    length=len(contours)
    if 0<length<=60:
        contours.sort(key=lambda x: cv2.contourArea(x),reverse=True)
        
        if cv2.contourArea(contours[0])>=3000:  #检测物体的最小面积
            sim_cout_0,sim_four_0=Similarity_cal(contours[0])
            
            if length>=2 and cv2.contourArea(contours[1])>=3000:
                sign_2=True
                sim_cout_1,sim_four_1=Similarity_cal(contours[1])

            sum_cout=sim_cout_0+sim_cout_1
            index_list=[sum_cout.index(x) for x in sum_cout if x<=0.24]  #相似度阈值

            if index_list:
                if sign_2 and len(index_list)>1:
                    sum_four=scfun.Normalization(sim_four_0+sim_four_1)
                    result=[sum_cout[x]*sum_four[x] for x in index_list]
                    min_index=index_list[result.index(min(result))]
                else:
                    min_index=index_list[0]
            
                if sign_2 and min_index>=5:
                    large_cout=contours[1]
                else:
                    large_cout=contours[0]
                sign=True

                return sign,large_cout

    return sign,0


#手势跟踪与识别
def Gestures_Detect(hand,sample_list,fourier_des_ls):
    ndefects=0
    
    sign,large_cout =Find_Contour(hand,sample_list,fourier_des_ls)
    if sign==False:
        ndefects=11  #返回contours为空的信息，只作调试用
        center=tuple([a//2 for a in reversed(hand.shape)])  #返回图像的中心坐标
        return hand,ndefects,center
    
    
    black2 = np.ones(hand.shape, np.uint8) #创建黑色幕布
    cv2.drawContours(black2,large_cout,-1,(255,255,255),2) #绘制白色轮廓
    cv2.imshow('large_cout',black2)
    
    hull=cv2.convexHull(large_cout,returnPoints=False)
    defects=cv2.convexityDefects(large_cout,hull)
    _,radius=cv2.minEnclosingCircle(large_cout)
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,_ =defects[i,0]
            sta=tuple(large_cout[s][0])
            end=tuple(large_cout[e][0])
            far=tuple(large_cout[f][0])
            B=scfun.Eucledian_Distance(sta,far)
            C=scfun.Eucledian_Distance(end,far)
            #过滤掉角边太短的角
            if B+C > radius:
                A=scfun.Eucledian_Distance(sta,end)  #底边
                angle=acos((B**2 + C**2 - A**2)/(2*B*C))

                if angle <= pi/2.5:
                    ndefects+=1
    else:
        ndefects=12
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
    for num in [0,1,2,3,4]:
        times=buf.count(num)
        if times>=int(length/2):
            return num
    return 13


def main():
    last_time=0
    R=70
    length=30
    ges_num_buf=[0]*length

    hand_contours_ls,fourier_des_ls =Create_Hand_Sample_Contour()  #加载手部样本轮廓
    
    cap=cv2.VideoCapture(0)  #启动摄像头
    model=cv2.createBackgroundSubtractorKNN(history=300,detectShadows=False)  #背景移除KNN模型
    
    while(1):
        
        _,frame =cap.read()
        
        start=clock()
        #frame =cv2.bilateralFilter(frame,5,50,100)  #双边滤波
        xframe =Remove_Background(model,frame)
        hand =Bodyskin_Detect_Otsu(xframe)
        
        afhand,ges_num,hand_center=Gestures_Detect(hand,hand_contours_ls,fourier_des_ls)
        real_ges_num=Ges_Num_Detect(ges_num,ges_num_buf,length=length)

        frame_center=tuple([a//2 for a in reversed(hand.shape)])  #返回图像的中心坐标
        cv2.line(afhand,hand_center,frame_center,(255,0,0),3)
        cv2.circle(afhand,frame_center,R,(255,255,0),1)
        if scfun.Eucledian_Distance(hand_center,frame_center)>=R:
            angle=scfun.Azimuth_Angle(hand_center,origin=frame_center)
            cv2.putText(afhand,str(real_ges_num+1),(0,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        else:
            angle=0
        
        cv2.imshow('koishi',afhand)
        cv2.imshow('origin',xframe)
        
        finish=clock()
        this_time,last_time=scfun.Filter(finish-start,last_time,rate=0.2)
        print('{:.5f}  {}  {:.5f}  {}  {:.5f}'.format(ges_num, real_ges_num+1,
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

