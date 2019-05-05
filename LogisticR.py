#logistic regression  on insurance dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


df = pd.read_csv("Insurance_dataset.csv")
print("-----ORIGINAL DATASET-----")
print(df.head(20))

#plt.scatter(df.age,df.pred,marker='+',color='green')
#plt.show()

print("Shape of dataset is: ")
print(df.shape)

X_train,x_test,y_train,y_test=train_test_split(df[['age']],df.pred,test_size=0.2)

print("---X_train---")
print(X_train)
print("---x_test---")
print(x_test)
print("---y_train---")
print(y_train)
print("---y_test---")
print(y_test)

model=LogisticRegression()
#returns Logistic regression model
model.fit(X_train,y_train)

print(model.predict(x_test))

print(model.predict([[30]]))


print(model.score(x_test,y_test))





