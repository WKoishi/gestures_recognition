#secondary function  次级函数

from math import sqrt,atan2,pi,exp
import numpy as np
from time import clock

#生成的傅里叶描述子个数
MIN_DESCRIPTOR = 32

#一阶低通滤波器
def Filter(val,last_val,rate):
    baro = rate*val + (1-rate)*last_val
    last_val=baro
    return baro,last_val

#计算x,y两点的欧氏距离
def Eucledian_Distance(x,y):
    return sqrt(sum([(a-b)**2 for a,b in zip(x,y)]))

#归一化函数
def Normalization(data):
    _range = np.max(data)
    return data / _range

#计算方位角
#value为输出入的坐标，origin为原点坐标，默认为(0,0)
#输出的角度以原点水平向右为x轴，逆时针为正，顺时针为负
def Azimuth_Angle(value,origin=(0,0)):
    return atan2(value[1]-origin[1],value[0]-origin[0])*(180/pi)

#Sigmoid函数
def Sigmoid(x):
    return 1/(1+exp(-x))

def Softmax(ls):
    return np.exp(ls)/np.sum(np.exp(ls))

'''
def Point_Fill(contour):
    min_dis=4
    new_cout=[]
    contour=list(contour)
    for coor in zip(contour, contour[1:] + contour[:1]):
        x=coor[0][0]-coor[1][0]
        y=coor[0][1]-coor[1][1]
        new_cout.append(list(coor[0]))

        if not y and abs(x)>=2*min_dis:
            num = abs(int(x/min_dis))-1
            d= -min_dis if x>0 else min_dis
            for k in range(num):
                new_cout.append([coor[0][0]+(k+1)*d, coor[0][1]])
        
        elif not x and abs(y)>=2*min_dis:
            num = abs(int(y/min_dis))-1
            d= -min_dis if y>0 else min_dis
            for k in range(num):
                new_cout.append([coor[0][0], coor[0][1]+(k+1)*d])
    
    new_cout=np.array(new_cout)
    return new_cout
'''

#根据轮廓计算傅里叶描述子
#从傅里叶变换结果中取出MIN_DESCRIPTOR个描述点
def Fourier_Descriptor(contour,Normalize=False):

    contours_complex = np.empty(contour.shape[:-1], dtype=complex)
    contours_complex.real = contour[:,0]  #横坐标作为实数部分
    contours_complex.imag = contour[:,1]  #纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)  #进行傅里叶变换

    if len(fourier_result)>MIN_DESCRIPTOR:
        #截短傅里叶描述子
        half=int(MIN_DESCRIPTOR/2)
        descriptors=np.append(fourier_result[:half],fourier_result[-half:])
    else:
        descriptors = fourier_result

    if Normalize:
        M=abs(descriptors[1])
        reconstruct=[abs(x)/M for x in descriptors[1:-1]]
        return reconstruct

    return descriptors

#根据傅里叶描述子重新构建轮廓
def Fourier_Contour(descriptors,img_shape):
    
    reconstruct = np.fft.ifft(descriptors)
    reconstruct = np.array([reconstruct.real,reconstruct.imag])
    reconstruct = np.transpose(reconstruct)
    reconstruct = np.expand_dims(reconstruct, axis = 1)

    reconstruct *= img_shape[0] / reconstruct.max()
    reconstruct = reconstruct.astype(np.int32, copy = False)

    return reconstruct

'''
#对傅里叶描述子进行归一化
def Fourier_Normalization(descriptors):
    M=abs(descriptors[1])
    reconstruct=np.array([abs(x)/M for x in descriptors[1:]])
#    reconstruct = np.array([reconstruct.real,reconstruct.imag])
#    reconstruct=np.transpose(reconstruct)
#    reconstruct=Normalization(reconstruct)

    return reconstruct
'''


