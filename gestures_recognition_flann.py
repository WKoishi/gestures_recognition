#FLANN匹配器

from cv2 import cv2
import matplotlib.pyplot as plt
from time import clock

img1=cv2.imread('hand_BGR_5_target.png')
img2=cv2.imread('hand_BGR_5.png')

img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

FLANN_INDEX_LSH=6
indexParams=dict(algorithm=FLANN_INDEX_LSH, 
                 table_number = 6, #12
                 key_size = 12,    #20
                 multi_probe_level = 1)#2
searchParams=dict(checks=100)

orb=cv2.ORB_create()

sta=clock()
kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)

flann=cv2.FlannBasedMatcher(indexParams,searchParams)
matches=flann.knnMatch(des1,des2,k=2)
matchesMask=[[0,0] for i in range(len(matches))]

for i in range(len(matches)):
    if len(matches[i])>=2:
        (m,n)=matches[i]
        if m.distance<0.7*n.distance:
            matchesMask[i]=[1,0]
fin=clock()

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

print(fin-sta)

cv2.imshow('sqwd',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

