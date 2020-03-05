import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

alldata = pd.read_csv(r'C:\Users\Sarath\Desktop\DS Hackathon\Files\UCI Repository\heart.csv')
alldata = np.asarray(alldata)
X = alldata[:, 0:13]
y = alldata[:, 13:]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
np.set_printoptions(precision=3)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=3)

tree = MLPClassifier(alpha=1)
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.8301886792452831 (Changes in the range of 80 and 85)
print(accuracy_score(Y_test, Y2)) #0.8351648351648352


