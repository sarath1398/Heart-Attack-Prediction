import sklearn
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler

alldata = pd.read_csv(r'C:\Users\Sarath\Desktop\DS Hackathon\Files\UCI Repository\heart.csv')
alldata=np.asarray(alldata)
X=alldata[:,0:13]
y=alldata[:,13:]
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
np.set_printoptions(precision=3)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=18)

classifier=QuadraticDiscriminantAnalysis()
classifier=classifier.fit(X_train,Y_train)
train_acc=classifier.predict(X_train)
test_acc=classifier.predict(X_test)
print(accuracy_score(Y_train, train_acc)) #0.8820754716981132
print(accuracy_score(Y_test, test_acc)) #0.8681318681318682

classifier=LinearDiscriminantAnalysis()
classifier=classifier.fit(X_train,Y_train)
train_acc=classifier.predict(X_train)
test_acc=classifier.predict(X_test)
print(accuracy_score(Y_train, train_acc)) #0.8584905660377359
print(accuracy_score(Y_test, test_acc)) #0.8461538461538461 




