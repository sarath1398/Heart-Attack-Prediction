import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

alldata = pd.read_csv(r'C:\Users\Sarath\Desktop\DS Hackathon\Files\UCI Repository\heart.csv')
alldata = np.asarray(alldata)
X = alldata[:, 0:13]
y = alldata[:, 13:]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
np.set_printoptions(precision=3)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=3)

log = LogisticRegression()
log = log.fit(X_train, Y_train)
Y1 = log.predict(X_train)
Y2 = log.predict(X_test)
print(accuracy_score(Y_train, Y1))
print(accuracy_score(Y_test, Y2))

