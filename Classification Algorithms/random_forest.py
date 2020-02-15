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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

f = pd.read_csv(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\heart.csv")
f = np.asarray(f)
x = f[:, 0:13]
y = f[:, 13]

tree = AdaBoostClassifier()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))  #91.75%

tree = BaggingClassifier()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #100%

tree = ExtraTreesClassifier()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #100%

tree = GradientBoostingClassifier()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #99.00%

tree = RandomForestClassifier() 
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))  # 100%
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
