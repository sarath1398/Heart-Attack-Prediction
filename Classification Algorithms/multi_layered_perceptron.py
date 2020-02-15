import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

f = pd.read_csv(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\heart.csv")
f = np.asarray(f)
x = f[:, 0:13]
y = f[:, 13]

tree = MLPClassifier(alpha=0.1)
tree = tree.fit(x, y)
y1 = tree.predict(x)
print(accuracy_score(y, y1))  # Has range between 83 and 87 with a max value of of 87.7% 
'''
tree = MLPRegressor()
tree = tree.fit(x, y)
y1 = tree.predict(x)

print(accuracy_score(y, y1))
'''