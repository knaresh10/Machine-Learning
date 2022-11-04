import pandas as pd
import numpy as np
a=['buying','maint','doors','persons','lug_boot','safety','class']
data=pd.read_csv(r'cars - 1.csv',names=a)

data.shape
data.size
data.info()
data.describe()

data.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['buying']=le.fit_transform(data['buying'])
data['maint']=le.fit_transform(data['maint'])
data['lug_boot']=le.fit_transform(data['lug_boot'])
data['safety']=le.fit_transform(data['safety'])

data['persons'].replace(to_replace='more',value=6,
                        inplace=True)

data['doors'].replace(to_replace='5more',value=5,
                        inplace=True)

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(
    x,y,test_size=0.2,random_state=5)


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred))

print(model.predict([[3,2,2,4,1,
                     2]]))