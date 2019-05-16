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

from sklearn.model_selection import KFold
kf=KFold(n_splits=3)
print(kf)
#Now my k-Fold is ready

for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):    
#in brackets of kf.split() you need to send dataset
#Now I will just print train index and test index
    print(train_index,test_index)
'''[3 4 5 6 7 8] [0 1 2]
    [0 1 2 6 7 8] [3 4 5]
    [0 1 2 3 4 5] [6 7 8]'''
#kf.split return an iterator that iterator will return train and test index for each of the iterations,so it
#divided into three folds i.e 3 each,the first iteration will use one fold testing which is {0 1 2} and remaining
#two folds for training i.e {3 4 5 6 7 8}and it repeats the procedure

'''We are going to use this K-Fold to our digits dataset'''
#i am taking one generic function which will take model as an input then  X_train,x_test,y_train,y_test.

def get_score(model,X_train,x_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(x_test,y_test)

print(get_score(lr,X_train,x_test,y_train,y_test))
print(get_score(sv,X_train,x_test,y_train,y_test))
print(get_score(rf,X_train,x_test,y_train,y_test))


from sklearn.model_selection import StratifiedKFold
#stratifiedKFold is similer to k fold but it is little better in a way that,when you are seperating out your
#folds it will divide each of the classification categories in a uniform way,this could be very helpfull.Imagine
#you are creating three folds eg.in iris dataset in 3 folds 2 folds have very simpiler type of flowers and third
#set will have very different type of flower,this will create problem thus by using strtifiedKfold is better.
folds=StratifiedKFold(n_splits=3)
score_l=[]
score_svm=[]
score_rf=[]

for train_index,test_index in kf.split(digits.data):
    X_train,x_test,y_train,y_test=digits.data[train_index],digits.data[test_index],\
                                   digits.target[train_index],digits.target[test_index]
#as our folds are 3 so this for loop will run three times every time taking different
#X_train,x_test,y_train,y_test and measure performance of our model and will append the scores in above lists
    score_l.append(get_score(lr,X_train,x_test,y_train,y_test))
    score_svm.append(get_score(sv,X_train,x_test,y_train,y_test))
    score_rf.append(get_score(rf,X_train,x_test,y_train,y_test))

print("Logistic regression score is: ",score_l)
#[0.8964941569282137, 0.9515859766277128, 0.9115191986644408]
print("SVM score is ",score_svm)
#[0.41068447412353926, 0.41569282136894825, 0.4273789649415693]
print("Random forest score is ",score_rf)
#[0.8781302170283807, 0.9165275459098498, 0.8864774624373957]
#You can take the average of these three scores and according to that decide the which model is performing better.
#Logistic regression model might perform best

#we can directly do the above things very easily as follows
from sklearn.model_selection import cross_val_score
print(cross_val_score(LogisticRegression(),digits.data,digits.target))
print(cross_val_score(SVC(),digits.data,digits.target))
print(cross_val_score(RandomForestClassifier(),digits.data,digits.target))





