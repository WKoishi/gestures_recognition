#secondary function  次级函数

from math import sqrt,atan2,pi,exp
import numpy as np

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

#计算归一化后的傅里叶描述子的欧氏距离
def Fourier_Descriptor_Similarity(A,B):
    summary=0
    for i in range(len(B)):
        summary+=Eucledian_Distance(A[i],B[i])
    #summary/=len(B)
    return summary

#归一化函数
def Normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

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


#根据轮廓计算傅里叶描述子
#从傅里叶变换结果中取出MIN_DESCRIPTOR个描述点
def Fourier_Descriptor(contour):
    contours_complex = np.empty(contour.shape[:-1], dtype=complex)
    contours_complex.real = contour[:,0]  #横坐标作为实数部分
    contours_complex.imag = contour[:,1]  #纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)  #进行傅里叶变换

    #截短傅里叶描述子
    descriptors = np.fft.fftshift(fourier_result)
    #取中间的MIN_DESCRIPTOR项描述子
    center_index = int(len(descriptors) / 2)
    low, high = center_index - int(MIN_DESCRIPTOR / 2), center_index + int(MIN_DESCRIPTOR / 2)
    descriptors = descriptors[low:high]
    descriptors = np.fft.ifftshift(descriptors)

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

#对傅里叶描述子进行归一化
def Fourier_Normalization(descriptors):
    M=abs(descriptors[1])
    reconstruct=np.array([abs(x)/M for x in descriptors[1:]])
#    reconstruct = np.array([reconstruct.real,reconstruct.imag])
#    reconstruct=np.transpose(reconstruct)
#    reconstruct=Normalization(reconstruct)

    return reconstruct



