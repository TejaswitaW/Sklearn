#time series analysis in pandas
from matplotlib import pyplot as plt
import pandas as pd
##df=pd.read_csv("aapl.csv")
##print(df.head())
##print("To know the type of date column it is sring")
##print(type(df.Date[0]))
##df=pd.read_csv("aapl.csv",parse_dates=['Date'],)
##print(df.head())
##print("To know the type of date column,now it is timestamp")
##print(type(df.Date[0]))
#now we are going to make this date column as index of the dataframe now it is
#integer
df=pd.read_csv("aapl.csv",parse_dates=['Date'],index_col="Date")
print(df.head())
print("Only getting May 2018 data")
print(df["2018-05"])
print("Finding average price of stock in the month of May")
#First we have to do  is closing price mean
print(df.index)
print(df["2018-05"].Close.mean())
#print("Retriving price on any particular date")
#print(df["2018-05-22"])--->gives error
print("Following gives data values in particular range")
print(df["2018-05-01":"2018-05-22"])
print("I want monthly frequency")
print(df.Close.resample('M').mean())
#df.Close.resample('M').mean().plot()#monthly frequency
df.Close.resample('W').mean().plot()#weekly frequency
plt.show()
