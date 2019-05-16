'''Implementation of Random Forset algorithm using Sklearn'''
'''Author:Ms.Tejaswita'''
'''Dataset:Sklearn digits_dataset'''
import pandas as pd
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
#importing required libraries

digits=load_digits()
print(dir(digits))
#['DESCR', 'data', 'images', 'target', 'target_names']

plt.gray()

##for i in range(4):
##    plt.matshow(digits.images[i])
##    plt.show()#8X8 array,each image is of 64 integer

df=pd.DataFrame(digits.data)
print(df.head())

print(digits.target)
#i am going to append this target variable in my dataframe
#as follows you can make new column in pandas dataframe
df['target']=digits.target
print(df.head()) 
'''  0    1    2     3     4     5   ...      59    60    61   62   63  target
0  0.0  0.0  5.0  13.0   9.0   1.0   ...    13.0  10.0   0.0  0.0  0.0       0
1  0.0  0.0  0.0  12.0  13.0   5.0   ...    11.0  16.0  10.0  0.0  0.0       1
2  0.0  0.0  0.0   4.0  15.0  12.0   ...     3.0  11.0  16.0  9.0  0.0       2
3  0.0  0.0  7.0  15.0  13.0   1.0   ...    13.0  13.0   9.0  0.0  0.0       3
4  0.0  0.0  0.0   1.0  11.0   0.0   ...     2.0  16.0   4.0  0.0  0.0       4'''
#if we consider row 0 ,all its 64 samples are mapped to 0..same mapping for remaining.
#now I will do train test split
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)
print(len(X_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
'''1437
360
1437
360'''
#Now I am going to use random forest classifier to train my model
#Ensemble term is used when you are using multiple algorithms to predict the outcome
#here we are building multiple decision trees and taking majority vote to come up with our final outcome,thus
#it is called emnsemble
from sklearn.ensemble import RandomForestClassifier
#n_estimator is the number of trees 
model=RandomForestClassifier()
model.fit(X_train,y_train)
print(model.score(x_test,y_test))
#now I want to see the confusion matrix to see how my model perform
y_predicted=model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
#y_test is truth and y_predicted is predicted from x_test
print(cm)
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


