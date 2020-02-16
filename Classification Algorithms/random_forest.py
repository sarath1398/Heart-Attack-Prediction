import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding
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

tree = AdaBoostClassifier()
tree = tree.fit(X, Y)
Y1 = tree.predict(X)
Y2 = tree.predict(x)
# Has range between 83 and 87 with a max value of of 87.7%
print(accuracy_score(Y, Y1))
print(accuracy_score(y, Y2))

tree = BaggingClassifier()
tree = tree.fit(X, Y)
Y1 = tree.predict(X)
Y2 = tree.predict(x)
# Has range between 83 and 87 with a max value of of 87.7%
print(accuracy_score(Y, Y1))
print(accuracy_score(y, Y2))

tree = ExtraTreesClassifier()
tree = tree.fit(X, Y)
Y1 = tree.predict(X)
Y2 = tree.predict(x)
# Has range between 83 and 87 with a max value of of 87.7%
print(accuracy_score(Y, Y1))
print(accuracy_score(y, Y2))

tree = GradientBoostingClassifier()
tree = tree.fit(X, Y)
Y1 = tree.predict(X)
Y2 = tree.predict(x)
# Has range between 83 and 87 with a max value of of 87.7%
print(accuracy_score(Y, Y1))
print(accuracy_score(y, Y2))

tree = RandomForestClassifier() 
tree = tree.fit(X, Y)
Y1 = tree.predict(X)
Y2 = tree.predict(x)
# Has range between 83 and 87 with a max value of of 87.7%
print(accuracy_score(Y, Y1))
print(accuracy_score(y, Y2))
'''
tree = StackingClassifier()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))

tree = VotingClassifier()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))

tree = AdaBoostRegressor()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))

tree = BaggingRegressor()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))

tree = ExtraTreesRegressor()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))

tree = GradientBoostingRegressor()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))

tree = RandomForestRegressor()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))

tree = StackingRegressor()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))

tree = VotingRegressor()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))
'''
