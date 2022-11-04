# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:56:37 2022

@author: Naresh
"""

import pandas as pd
df1 = pd.read_excel(r"./kmeans1.xlsx")
df = df1.drop(['ID Tag','Model','Department'], axis = 1)
print(df.head())
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
km.fit(df)
x = km.fit_predict(df)
df1['cluster'] = x
df = df1.sort_values(['cluster'])
print(df)

df.to_csv('kmeansData.csv')