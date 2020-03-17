import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from math import sqrt,atan2,pi,exp
from time import clock
from collections import Counter
from fourierDescriptor import fourierDesciptor,reconstruct
from scipy import fftpack

def Eucledian_Distance(x,y):
    return sqrt(sum([(a-b)**2 for a,b in zip(x,y)]))

def Manhattan_Distance(x,y):
    return sum([abs(a-b) for a,b in zip(x,y)])

def Ges_Num_Detect(new,buf,length):
    for i in range(length-1):
        buf[i]=buf[i+1]
    buf[length-1]=new
    obj=Counter(buf)
    most=obj.most_common(1)
    if most[0][1]>=int(length/2):
        return most[0][0]
    else:
        return 12

def Test(new,buf,length):
    for i in range(length-1):
        buf[i]=buf[i+1]
    buf[length-1]=new
    for num in [0,1,2,3,4]:
        times=buf.count(num)
        if times>=15:
            return num
    return 12

def Avg_Filter(new,buf,length):
    for i in range(length-1):
        buf[i]=buf[i+1]
    buf[length-1]=new
    max_id=buf.index(max(buf))
    min_id=buf.index(min(buf))
    buf[max_id]=0
    buf[min_id]=0
    return sum(buf)/(length-2)

#计算方位角
#value为输入的坐标，origin为原点坐标，默认为(0,0)
#输出的角度以原点水平向右为x轴，逆时针为正，顺时针为负
def Azimuth_Angle(value,origin=(0,0)):
    #coor=[a-b for a,b in zip(value,origin)]
    return atan2(value[1]-origin[1],value[0]-origin[0])*(180/pi)

def Softmax(ls):
    x_exp=[exp(x) for x in ls]
    tmp=sum(x_exp)
    return [x/tmp for x in x_exp]

def test(ls):
    return np.exp(ls)/np.sum(np.exp(ls))

'''
start=clock()
hi=Ges_Num_Detect(13,ls,30)
finish=clock()
print(hi,finish-start)

start=clock()
ji=Test(13,ls,30)
finish=clock()
print(ji,finish-start)
'''

def deskew(img,shape):
    SZ=int((shape[0]+shape[1])/2)
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img

ls=np.zeros((2500,2),dtype=np.float64)
for i,_ in enumerate(ls):
    for j,_ in enumerate(ls[i]):
        ls[i,j]=5*sqrt(i)-80*j

df=[]
for i in range(10):
    df.append(i+1)

contours_complex = np.empty(ls.shape[:-1], dtype=complex)
contours_complex.real = ls[:,0]  #横坐标作为实数部分
contours_complex.imag = ls[:,1]  #纵坐标作为虚数部分

start=clock()
wcxe=np.ones((11,11),dtype=np.uint8)

finish=clock()
print(finish-start)





