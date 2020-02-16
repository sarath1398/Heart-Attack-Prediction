import numpy as np 
import pandas as pd 
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import NuSVC
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

fitdata = pd.read_excel(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\UCI Repository\fitdata.xlsx")
predictdata = pd.read_excel(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\UCI Repository\predictiondata.xlsx")
fitdata = np.asarray(fitdata)
predictdata = np.asarray(predictdata)
X = fitdata[:, 0:13]
Y = fitdata[:, 13]
x = predictdata[:, 0:13]
y = predictdata[:, 13]
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
x = scaler.fit_transform(x)
np.set_printoptions(precision=3)

tree = SVC()
tree = tree.fit(X, Y)
Y1 = tree.predict(X)
Y2 = tree.predict(x)
# Has range between 83 and 87 with a max value of of 87.7%
print(accuracy_score(Y, Y1))
print(accuracy_score(y, Y2))
'''
tree = SVR()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
tree = NuSVC()
tree = tree.fit(X, Y)
Y1 = tree.predict(X)
Y2 = tree.predict(x)
# Has range between 83 and 87 with a max value of of 87.7%
print(accuracy_score(Y, Y1))
print(accuracy_score(y, Y2))
'''
tree = NuSVR()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
tree = LinearSVC()
tree = tree.fit(X, Y)
Y1 = tree.predict(X)
Y2 = tree.predict(x)
# Has range between 83 and 87 with a max value of of 87.7%
print(accuracy_score(Y, Y1))
print(accuracy_score(y, Y2))
'''
tree = LinearSVR()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
tree = OneClassSVM()
tree = tree.fit(X, Y)
Y1 = tree.predict(X)
Y2 = tree.predict(x)
# Has range between 83 and 87 with a max value of of 87.7%
print(accuracy_score(Y, Y1))
print(accuracy_score(y, Y2))

