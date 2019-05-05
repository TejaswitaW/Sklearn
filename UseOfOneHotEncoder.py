#use of dummy variables,use of one hot encoder


'''Use of dummy variable from pandas(One Hot Encoding'''
'''Author:Ms.Tejaswita'''

#imported required library
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("Town_dataset.csv")
print("------ORIGINAL DATASET------")
print(df.head(20))


#to use one hot encoder,you have to use lable encoding on the town coulmn,for that I will use lable encoder from
#sklearn.preprocessing and create the label class object as follows

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
#created lable encoder class object
#now we are going to use it on our original dataframe
#dfle is resultant dataframe

#lets create new dataframe
dfle=df
dfle.town=le.fit_transform(dfle.town)#means it takes my label column as input and returns the label(int values)
dfle # this is label encoded will look as follows
print("----Label Encoded------")
print(dfle)
#I will get my town categories converted to integer numbers

#Next I have i have to create x
x=df[['town','area']].values#values because this time i want x as array not as dataframe,when you call values on
#dataframe then you will get 2D array
print("-----2D Array of X-----")
print(x)

#now my y
y=dfle.price
print("-----Y-----")
print(y)

#we need to create dummy variable columns here import one hot encoder from sklearn.preprocessing
#I will create object of this class first
ohe=OneHotEncoder(categorical_features=[0])#we need 0th column

x=ohe.fit_transform(x).toarray()#this is my x now
print("After applying one hot encoder")
print(x)

