# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:19:22 2022

@author: Naresh
"""

import pandas as pd

colHeader = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv(r'boston.csv',names = colHeader)

data = data.dropna()

data.shape
data.size
data.head()
data.info()
b = data.describe()

x = data.iloc[:,:-1].values 
y = data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.3, random_state=88)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import mean_squared_error
import math
print(math.sqrt(mean_squared_error(ytest,ypred)))


print(model.predict([xtrain[0]]))


