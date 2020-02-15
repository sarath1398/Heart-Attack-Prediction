import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import BaseDecisionTree
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

f = pd.read_csv(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\heart.csv")
f = np.asarray(f)
x = f[:,0:13]
y = f[:,13]

tree=DecisionTreeClassifier()
tree=tree.fit(x,y)
y1=tree.predict(x)
print(accuracy_score(y,y1)) #100% Accuracy and that's impossible 

tree=DecisionTreeRegressor()
tree=tree.fit(x,y)
y1=tree.predict(x)
print(accuracy_score(y, y1))  # 100% Accuracy again..There's something definitely wrong with the code!

tree=ExtraTreeClassifier()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #Hattrick

tree=ExtraTreeRegressor()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #Hattrick+1
'''
tree=BaseDecisionTree()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
