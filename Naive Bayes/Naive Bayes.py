###Naive bayes

import pandas as pd

###import dataset
names=[]
for i in 'abcdefghijklmno':
    names.append(i)


data=pd.read_csv(r'income.csv',names=names)


###replace with particular value
data['b']=data['b'].replace(to_replace=' ?',value=' Private')

data['g']=data['g'].replace(to_replace=' ?',value=' Prof-specialty')

data['n']=data['n'].replace(to_replace=' ?',value=' United-States')


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
##

for i in 'bdfghijn':
    data[i]=le.fit_transform(data[i])

data['o']=le.fit_transform(data['o'])

##
##
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


##
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=6)


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)