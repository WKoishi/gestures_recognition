import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from math import sqrt,atan2,pi,exp
from time import perf_counter
from collections import Counter
from scipy import fftpack

def Eucledian_Distance(x,y):
    return sqrt(sum([(a-b)**2 for a,b in zip(x,y)]))

def Eucledian_Distance_Mat(x,y):
    if x.shape==y.shape:
        return np.sqrt(np.dot(x,x)-2*np.dot(x,y)+np.dot(y,y))
    else:
        return 0XFF

def ED_test(x,y):
    return np.linalg.norm(x - y)

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

#归一化函数
def Normalization(data):
    _range = abs(np.max(data))
    return data / _range

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

def OTSU(img,sta,fin):

    hist = cv2.calcHist([img],[0],None,[fin-sta],[sta,fin])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()

    bins = np.arange(fin-sta)

    fn_min = np.inf
    thresh = -1

    for i in range(1,fin-sta):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[fin-sta-1]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights

        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    thresh+=(sta-1)

    return thresh



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

ls=[4,6,3,11,0.5,0.7,22,34,12,7,8,10]
index=[3,6,15]
avg=sum(index)/len(index)
print(avg)
'''

'''
sta=clock()
for _ in range(5):
    dqx=Eucledian_Distance_Mat(A,B)
fin=clock()
print(dqx)
print(fin-sta)
'''

cap=cv2.VideoCapture(0)

while(1):

    _,img=cap.read()
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img,(5,5),0)
    '''
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()

    bins = np.arange(256)

    fn_min = np.inf
    thresh = -1

    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights

        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    '''
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    print(fin-sta)

    cv2.imshow('cxw',img)

    key=cv2.waitKey(2)&0XFF
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()








