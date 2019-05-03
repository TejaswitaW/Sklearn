#multivariate linear regression
#importing reuired libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
import math


df=pd.read_csv("HomePriceM.csv")#read dataset csv file
print(df.head(11))#observing data points

#calculating median of number of bedrooms
median_b=math.floor(df.bedrooms.median())
print("Median of number of bedrooms is: ",median_b)

#filling median value at NaN position of bedrooms column 
df.bedrooms=df.bedrooms.fillna(median_b)
print(df.bedrooms)
print(df.head(11))

#applying linear regression model,creating object
reg=linear_model.LinearRegression()

#training our model
r=reg.fit(df[['area','bedrooms','age']],df.price)

#coefficient
print("Coefficient of this model is: ",r.coef_)
#intercept
print("Intercept of this model is: ",r.intercept_)
#prediction for new values
p=r.predict([[3000,4,15]])
print("Predicted price by the model is: ",p)
