import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomTreesEmbedding
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

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=2)

tree = AdaBoostClassifier()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.9386792452830188
print(accuracy_score(Y_test, Y2)) #0.8681318681318682

tree = BaggingClassifier()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.9952830188679245
print(accuracy_score(Y_test, Y2)) #0.8461538461538461

tree = ExtraTreesClassifier()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #1.0
print(accuracy_score(Y_test, Y2)) #0.8791208791208791

tree = GradientBoostingClassifier()
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.9952830188679245
print(accuracy_score(Y_test, Y2)) #0.8791208791208791

tree = RandomForestClassifier() 
tree = tree.fit(X_train, Y_train)
Y1 = tree.predict(X_train)
Y2 = tree.predict(X_test)
print(accuracy_score(Y_train, Y1)) #1.0
print(accuracy_score(Y_test, Y2)) #0.8791208791208791
