#use of stacking in pandas
import pandas as pd
df=pd.read_excel("Stock_Data.xlsx",header=[0,1])
print("Dataframe before reshaping")
print(df)
print("After stacking")
print(df.stack())
print("After taking level 0")
df1=df.stack(level=0)
print(df1)
print("After reverse transformation I get my orginal dataframe")
print(df1.unstack())
