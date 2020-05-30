from keras.models import load_model
import numpy as np

Hudata_mean=np.array([2.24416476e-01, 1.28205266e-02, 1.58525734e-03, 4.83401160e-04,
       6.70529236e-07, 5.48527875e-05, 1.56711286e-09])

Hudata_std=np.array([2.88563950e-02, 1.12261230e-02, 2.22427169e-03, 6.76677222e-04,
       2.78176258e-06, 1.13574967e-04, 1.12484895e-06])

model=load_model("gestures_recognition_Hu_F.h5")

def Predict(Hu_array):

    if len(Hu_array.shape)==1:
        Hu_array=np.expand_dims(Hu_array,axis=0)

    Hu_array-=Hudata_mean
    Hu_array/=Hudata_std


    result=model.predict(Hu_array)

    return result