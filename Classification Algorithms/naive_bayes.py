import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BaseDiscreteNB
from sklearn.naive_bayes import BaseNB
from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.naive_bayes import _BaseNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
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

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=14)

#BernoulliNB

nvb = BernoulliNB()
nvb = nvb.fit(X_train, Y_train)
Y1 = nvb.predict(X_train)
Y2 = nvb.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.8490566037735849
print(accuracy_score(Y_test, Y2)) #0.8131868131868132


#ComplementNB

nvb = ComplementNB()
nvb = nvb.fit(X_train, Y_train)
Y1 = nvb.predict(X_train)
Y2 = nvb.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.8018867924528302
print(accuracy_score(Y_test, Y2)) #0.7912087912087912

#MultinomialNB

nvb = MultinomialNB()
nvb = nvb.fit(X_train, Y_train)
Y1 = nvb.predict(X_train)
Y2 = nvb.predict(X_test)
print(accuracy_score(Y_train, Y1)) #0.8113207547169812
print(accuracy_score(Y_test, Y2)) #0.7692307692307693

#GaussianNB

nvb = GaussianNB()
nvb = nvb.fit(X_train, Y_train)
Y1 = nvb.predict(X_train)
Y2 = nvb.predict(X_test) 
print(accuracy_score(Y_train, Y1)) #0.8490566037735849
print(accuracy_score(Y_test, Y2)) #0.8241758241758241
