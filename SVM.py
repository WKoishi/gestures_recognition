import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,model_selection,svm

iris=datasets.load_iris()
x_train=iris.data
y_train=iris.target

x_train,x_test,y_train,y_test=model_selection.train_test_split(x_train,y_train,
                                                                test_size=0.25,
                                                                random_state=0,
                                                                stratify=y_train)
print(x_train[:1])
print(y_train[:1])

clas=svm.LinearSVC()
clas.fit(x_train,y_train)
print('各特征权重：%s,截距:%s'%(clas.coef_,clas.intercept_))
print("算法评分：%.2f" % clas.score(x_test,y_test))


print('xwd')



