'''SVM implementation in python using sklearn'''
'''Author:Ms.Tejaswita'''
import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
iris=load_iris()
#print(dir(iris))
#print(iris.feature_names)
df=pd.DataFrame(iris.data,columns=iris.feature_names)
#created my own dataframe
print(df.head())
df['target']=iris.target
#appended target in my own dataframe
#the possible values of target is 0,1 and 2
#what is mean by 0,1,2 means:
#['setosa' 'versicolor' 'virginica']
print(iris.target_names)
print(df.head())
#I want to see how many data points have 1 as target value
t=df[df.target==1].head()
#print(t)#shows versicolor class value
t1=df[df.target==2].head()
#print(t1)#shows verginica class value
df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
#print(df.head())#now above column is added to original data
#seperating these three species into different dataframe
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
#print(df0)
#print(df1)
#print(df2)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color="blue",marker="+")
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color="red",marker=".")
#plt.show()
#when we observe plot it shows clear classification
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color="blue",marker="+")
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color="red",marker=".")
#plt.show()
#when we train our algorithm we are going to use all 4 features
#also classification will also be done in all three species
#now training our model using sklearn
#first step is as usual use train test split to split our dataset into train and test dataset
from sklearn.model_selection import train_test_split
#our data has target column we want to remove that and the way you want to remove is that
X=df.drop(['target','flower_name'],axis='columns')
#print(X)
y=df.target
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#print(len(X_train))#120
#print(len(x_test))#30
#print(len(y_train))#120
#print(len(y_test))#30
from sklearn.svm import SVC
model=SVC(C=10,gamma='auto')
#C=0.1 is regularization parameter,increasing regularization is decreasing my score
#gamma and C are tunning parameters in SVM
m=model.fit(X_train,y_train)
print(model.score(x_test,y_test))

