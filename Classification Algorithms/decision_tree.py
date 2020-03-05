import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import BaseDecisionTree
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

alldata = pd.read_csv(r'C:\Users\Sarath\Desktop\DS Hackathon\Files\UCI Repository\heart.csv')
alldata = np.asarray(alldata)
X = alldata[:, 0:13]
y = alldata[:, 13:]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
np.set_printoptions(precision=3)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=2)

tree=DecisionTreeClassifier()
tree=tree.fit(X_train,Y_train)
Y1=tree.predict(X_train)
Y2=tree.predict(X_test)
print(accuracy_score(Y_train,Y1)) 
print(accuracy_score(Y_test,Y2)) 

tree=DecisionTreeRegressor()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1))
print(accuracy_score(Y_test, Y2))

tree=ExtraTreeClassifier()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1))
print(accuracy_score(Y_test, Y2))

tree=ExtraTreeRegressor()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1))
print(accuracy_score(Y_test, Y2))
'''
tree=BaseDecisionTree()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
