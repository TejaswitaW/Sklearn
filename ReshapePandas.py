#use of melt to reshpe in pandas
import pandas as pd

df=pd.read_csv("City_Weather_data.csv")
print("-----Orginal Dataframe---- ")
print(df)
print("----After transformation---")
df1=pd.melt(df,id_vars=["day"])
print(df1)
print("----New dataframe-----")
df2=pd.melt(df,id_vars=["day"],var_name="city",value_name="temperature")
print(df2)
