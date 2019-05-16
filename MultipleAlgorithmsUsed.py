from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
digits=load_digits()

from sklearn.model_selection import train_test_split

X_train,x_test,y_train,y_test=train_test_split(digits.data,\
                                               digits.target,test_size=0.3)
lr=LogisticRegression()
l=lr.fit(X_train,y_train)
ls=lr.score(x_test,y_test)
print("Score of Logistic Regression Model is: ",ls)
#Score of Logistic Regression Model is:  0.9481481481481482


sv=SVC()
s=sv.fit(X_train,y_train)
ss=sv.score(x_test,y_test)
print("Score of SVM Model is: ",ss)
#Score of SVM Model is:  0.4222222222222222


rf=RandomForestClassifier()
r=rf.fit(X_train,y_train)
rs=rf.score(x_test,y_test)
print("Score of Random Forest Model is: ",rs)
#Score of Random Forest Model is:  0.9462962962962963
'''If we run the program again then score value changes,why? beacuase when you run the program again then samples
in X_Train,x_test,y_train,y_test change,so score value change.The problem with train test split method is onece
you run the program you can not say it is the best model,you have to run it multiple times to get best model.So
try K-Fold'''


