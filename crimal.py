# -*- coding: utf-8 -*-
"""
Problem Statement:
2.Perform clustering for the crime data and identify the number of clusters            
  formed and draw inferences. Refer to crime_data.csv dataset.

objective:
    Identify the crime rate of particular state 

minimize:
    The crime rate

Maximize:
    security,quick action of police,use the technology at the find out 
    the crime patterns

Constraints:
    Politicians support the criminals, police force is inactive,lack of
    training for self defences in case of girls or childrens

@author: Sujata Mandale
"""

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from feature_engine.outliers import Winsorizer
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

#Load the Crime dataset
df=pd.read_csv(r'C:\\Users\\delll\\Desktop\\Python\\DataSets\\crime_data.csv ')

df.shape
#rows=50,columns=5

df.columns
'''
Name of Features| Description       |   Type     |          Relevance
1. Name           Name of states        Nominal      does not relevant to form clusters
2 .Murder         No. of Murders        continuous   relevant to form clusters
3.Assault         No. of Assault        discrete     relevant to form clusters
4.UrbanPop        Urban Population      discrete     does not relevant to form clusters
5.Rape            Rate of Rape          continuous   relevant to form clusters
'''
#Name-Nominal data
#Murder-continuous data
#Assault-discreate data
#UrbanPop-discreate data
#rape-continuous data
df.info()
#Name-odject with 50 records
#Murder-float with 50 records
#Assault-int with 50 records
#UrbanPop-int with 50 records
#rape-float with 50 records

df.isnull().sum()
#There is no null values

df.describe()

#Murder-mean-7.78800,std-4.35551,min-0.80000,max-17.40000
#Assault-mean-170.760000,std-83.337661,min-45.000000,max-337.000000
#UrbanPop-mean-65.540000,std-14.474763,min-32.000000,max-91.000000 
#rape-mean-21.232000,std-9.366385,min-7.300000,max-46.000000

sns.boxplot(df['Murder'])
sns.boxplot(df['Assault'])
sns.boxplot(df['UrbanPop'])
sns.boxplot(df['Rape'])

#there are no outlier in Murder,Assault and UrbanPop
#Rape columns has outlier

#To remove the outlier use winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Rape'])
df['Rape']=winsor.fit_transform(df[['Rape']])
sns.boxplot(df['Rape'])
#from boxplot we easily see that outliers removed..
# Bivariate analysis

sns.pairplot(df)
df.corr()
sns.heatmap(df.corr(),cmap='coolwarm',annot=True,fmt='.2f')
plt.show()

#from the heatmap and pairplot we can see that 
#1.Murder and Assault has positive strong linear relationship..
#2.Murder and Rape have positive weak linear relationship
#3.Assault and Rape have positive weak linear relationship
#sns.barplot(data=df,x='Murder',y='Assault')
#Skewness of columns
sns.distplot(df.Murder)
df['Murder'].skew()
#it is right skewwed data and skewness is 0.39
sns.distplot(df.Assault, kde=True)
df['Assault'].skew()
#it is right skewwed data and skewness is 0.23
sns.distplot(df.UrbanPop, kde=True)
df['UrbanPop'].skew()
#it is left skewwed data and skewness is -0.22
sns.distplot(df.Rape, kde=True)
df['Rape'].skew()
#it is right skewwed and skewness is 0.68

# Convert the Skewwed data to Normal distribution
target = np.log(df['Rape'])
print ('Skewness is', target.skew())
sns.distplot(target)
#If data is right skewwed then do log transform

target = np.square(df['UrbanPop'])
print ('Skewness is', target.skew())
sns.distplot(target)
#If data is left skewwed then do square transform

#drop columns Name which categorical data
df.drop(['Name'],inplace=True,axis=1)
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_fun(df.iloc[:,0:])
n=df_norm.describe()
sns.histplot(df_norm.Murder, kde=True)

z=linkage(df_norm,method='complete',metric='euclidean')
plt.figure(figsize=(15,8));
plt.title('Hierarchical clustering dendogram');
plt.xlabel('index');
plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()


h_complete=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity='euclidean').fit(df_norm)
h_complete.labels_
print(np.unique(h_complete.labels_))
#we formed the cluster [0,1,2,3]
cluster_labels=pd.Series(h_complete.labels_)

df['Cluster']=cluster_labels

df=df.iloc[:,[4,0,1,2,3]]
cluster_group=df.iloc[:,1:].groupby(df.Cluster).mean()
df.to_csv('Criminal.csv',encoding='utf8')
import os
os.getcwd()


