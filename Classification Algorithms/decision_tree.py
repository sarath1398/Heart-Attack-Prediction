''' Required Imports '''

import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import BaseDecisionTree
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

alldata = pd.read_csv(r'C:\Users\Sarath\Desktop\DS Hackathon\Files\UCI Repository\heart.csv') #Loaded the dataset
alldata = np.asarray(alldata) #Converted the CSV file into np array
X = alldata[:, 0:13] #Attribute data
y = alldata[:, 13:] #Target data
scaler = MinMaxScaler() #Scaler object to scale the data between 0 and 1
X = scaler.fit_transform(X) #Scaled the X data, Y is not scaled since Y is binary here!
np.set_printoptions(precision=3)

''' Train and Test splitted in 70-30 ratio! '''

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=2)

'''Implementing all the Decision Tree based Classifiers and Regressors for Prediction '''

tree=DecisionTreeClassifier()
tree=tree.fit(X_train,Y_train)
Y1=tree.predict(X_train)
Y2=tree.predict(X_test)
print(accuracy_score(Y_train,Y1))  #1.0
print(accuracy_score(Y_test,Y2))  #0.8791208791208791

tree=DecisionTreeRegressor()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #1.0
print(accuracy_score(Y_test, Y2)) #0.8571428571428571

tree=ExtraTreeClassifier()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #1.0
print(accuracy_score(Y_test, Y2)) #0.7472527472527473

tree=ExtraTreeRegressor()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #1.0
print(accuracy_score(Y_test, Y2)) #0.7912087912087912

''' 
tree=BaseDecisionTree()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''

