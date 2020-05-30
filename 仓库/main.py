from primaryFun import *
from otsu_threshold import Bodyskin_Detect_Otsu
from MLP_predict import Predict
from cv2 import cv2
import numpy as np
import scfun
from time import perf_counter

last_time=0
R=70
length=30
ges_num_buf=[0]*length

capture=cv2.VideoCapture(0)  #启动摄像头
capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while(1):

    ret,frame=capture.read()
    if ret==False:
        print("无法获取帧")
    
    sta=perf_counter()
    skin=Bodyskin_Detect_Otsu(frame)
    fin=perf_counter()

    print(fin-sta)

    contour_ls,convexHull,Hu_list=Find_Contour([skin])

    predict_res=Predict(Hu_list)

    index=[convexHull.index(x) for x in convexHull if x[0]>=1]
    if index:
        max_val,max_val_index = 0,0
        for i in index:
            smax=np.max(predict_res[i])
            if smax>max_val:
                max_val=smax
                max_val_index=i
        
        ndefect=np.where(predict_res[max_val_index]==max_val)[0]+1
    else:
        ndefect=12

    #print(ndefect)

    black=np.zeros(frame.shape[:2],dtype=np.uint8)
    if contour_ls:
        cv2.drawContours(black,contour_ls[0],-1,255,2)
    cv2.imshow('test',black)
    cv2.imshow('skin',skin)


    key=cv2.waitKey(1)&0XFF
    if key == 27:
        break  #按esc退出
    elif key == ord('q'):
        pass
    

capture.release()
cv2.destroyAllWindows()
