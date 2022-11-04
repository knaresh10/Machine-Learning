import pandas as pd
#import numpy as np
data = pd.read_csv(r'iris.csv')

data.shape # number of rows and no of columns
data.size  # number of elements

data.info()

data.describe()

#data = data.fillna(0.2)
#data.iloc[0:4,:]

# .values function will give you numpy array instead of pandas dataframe or series
x = data.iloc[:,:-1].values  # take independent variables( take all rows and only 1st 4 columns)

y = data.iloc[:,-1].values # take target variable (take class column)

# to split the data into training and testing data
from sklearn.model_selection import train_test_split

# data is split into 4 parts where train data contains 80% of original data and 20% data 
# random state is used to keep the random spliting of data same when doing multiple compliations 
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size= 0.2,random_state=4)

#model is created for knn
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors= 1)
model.fit(xtrain, ytrain)

# now try test the model
ypred = model.predict(xtest)

# to check the accuracy of the model using ypred and ytest
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, ypred)*100)

#to calculate percision using confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, ypred))



# now to check the future prediction

print(model.predict([[5.2,4.1,1.6,1.4]]))

print(model.predict([[7.1,3.9,2.0,1.2]]))



print(model.predict([[7.2,3,5.8,1.6]]))