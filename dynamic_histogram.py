from cv2 import cv2
import numpy as np

def DynamicHist(src,winname,section=[0,256],removeInZero=False,displaySize=(480,640)):
    hist = cv2.calcHist([src],[0],None,[section[1]-section[0]],section)
    hist=hist[:,0]
    if removeInZero:
        hist[0]=0

    binWidth=int(displaySize[1]/hist.shape[0])
    k=np.max(hist)/displaySize[0]
    hist=hist/k
    black=np.zeros((displaySize[0],hist.shape[0]*binWidth),np.uint8)

    i=0;j=0
    for x in np.nditer(hist):
        j+=binWidth
        black[:int(x),i:j]=200
        i+=binWidth

    p=int(displaySize[0]/20)
    scale=np.arange(0,black.shape[1],50*binWidth)
    black[-p:,scale]=0XFF
    
    black=cv2.flip(black,0)
    cv2.imshow(winname,black)





