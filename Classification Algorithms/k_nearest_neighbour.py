import numpy as np
import pandas as pd
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import RadiusNeighborsTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

f = pd.read_csv(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\heart.csv")
f = np.asarray(f)
x = f[:, 0:13]
y = f[:, 13]

tree = RadiusNeighborsClassifier()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))  #100%

tree = NearestCentroid()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1)) #65%
'''
tree = NearestNeighbors()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))  

tree = RadiusNeighborsTransformer()
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))  
'''

