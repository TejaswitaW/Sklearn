#use of crosstab in pandas using people dataset
import pandas as pd
df=pd.read_excel("People.xlsx")
print("----Original Dataframe----")
print(df)
#creating contigency table
df1=pd.crosstab(df.Nationality,df.Handedness)
print(df1)
print("Taking sex on x axis")
df1=pd.crosstab(df.Sex,df.Handedness)
print(df1)
#margins argument give total frequency
print("Taking margins argument")
df1=pd.crosstab(df.Sex,df.Handedness,margins=True)
print(df1)
print("Taking nationality also")
df1=pd.crosstab(df.Sex,[df.Handedness,df.Nationality],margins=True)
print(df1)
