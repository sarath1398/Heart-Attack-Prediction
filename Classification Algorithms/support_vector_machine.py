import numpy as np 
import pandas as pd 
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import NuSVC
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

tree = SVC()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train) 
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.8820754716981132
print(accuracy_score(Y_test, Y2)) #0.8901098901098901
'''
tree = SVR()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
tree = NuSVC()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.8820754716981132
print(accuracy_score(Y_test, Y2)) #0.8901098901098901
'''
tree = NuSVR()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
tree = LinearSVC()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.8443396226415094
print(accuracy_score(Y_test, Y2)) #0.8681318681318682
'''
tree = LinearSVR()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))

tree = OneClassSVM()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1))
print(accuracy_score(Y_test, Y2))
'''
