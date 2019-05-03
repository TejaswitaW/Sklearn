#house price prediction using sklearn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
#df=pd.read_csv("Project_Data.csv",sep='\s*,\s*',\
#                          header=0, encoding='ascii', engine='python')
df=pd.read_csv("HomePrice.csv")
reg=linear_model.LinearRegression()
#print("I am here 1")
#fitting means you are training your model
reg.fit(df[['area']],df.price)
#print("I am here 2")
#now we will do some prediction
p=reg.predict([[5000]])
#when wrote reg.predict(3300),Expected 2D array, got scalar array instead
print("Predicted value for given area is ",p)
##print(df.head(10))
##plt.xlabel("area(sq ft)")
##plt.ylabel("price(US$)")
##plt.scatter(df.area,df.price,color='red',marker="+")
##plt.show()
print("Coefficient of linear  regression is: ",reg.coef_)
print("Intercept of linear regression is: ",reg.intercept_)
print("Supplying csv file of areas")
d=pd.read_csv("areas.csv")
#print(d.head(10))
m=reg.predict(d)
d['prices']=m
print(d.head(13))
d.to_csv("House_price_pred.csv")


