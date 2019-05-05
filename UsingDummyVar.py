#use of dummy variables


'''Use of dummy variable from pandas(One Hot Encoding'''
'''Author:Ms.Tejaswita'''

#imported required library
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("Town_dataset.csv")
print("------ORIGINAL DATASET------")
print(df.head(20))

#following statement gives dummy values for town column
d=pd.get_dummies(df.town)
#print(d)#got dummy variables created 3 columns as we have 3 names of towns

#concatinating these dummy variables with our dataset dataframe
#pd.concat,connect two dataframes

m=pd.concat([df,d],axis='columns')
print("-------WITH DUMMY VARIABLES-------")
print(m)

#after we got dummy variables we need to drop town column in original dataset
#to avoid dummy variable trap we need to drop one column
print("---------FINAL----------")
f=m.drop(['town','WW'],axis='columns')
print(f)

#model object Creation
model=linear_model.LinearRegression()

#all the coulumns except price are independent variable i.e x,so your x is price,MT,RS and y is dependent varable
#i.e price so,gettting x and y as follows
print("-----X VALUE-----")
x=f.drop(['price'],axis='columns')
print(x)


print("-----X VALUE-----")
#y is price in dataset
y=f.price

#training our model
m=model.fit(x,y)
print(m)

print("Coefficient is : ",m.coef_)
print("Intercept is : ",m.intercept_)

print("Output for home price:2800 in RS is",m.predict([[2800,0,1]]))
#while predicting i need to give values to x as they look in x,we have to provide price,MT=0/1,RS=0/1
#see code as follows

print("Output for home price:2800 in WW is",m.predict([[2800,0,0]]))

print("Output for home price:2800 in MT is",m.predict([[2800,1,0]]))

print("My model score is: ",m.score(x,y))
#got 0.71
