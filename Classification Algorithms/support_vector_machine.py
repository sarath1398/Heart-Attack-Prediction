import numpy as np 
import pandas as pd 
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import NuSVC
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

f = pd.read_csv(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\heart.csv")
f = np.asarray(f)
x = f[:, 0:13]
y = f[:, 13]

tree = SVC()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #67.3% Accuracy
'''
tree = SVR()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
tree = NuSVC()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #84.48% accuracy
'''
tree = NuSVR()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
tree = LinearSVC()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #79.86% Accuracy 
'''
tree = LinearSVR()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
tree = OneClassSVM()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #That's a perfectly working model with 29.04% accuracy!

