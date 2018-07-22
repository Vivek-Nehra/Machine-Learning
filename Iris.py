import numpy as np
import pandas as pd
from sklearn import datasets,linear_model,svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

iris = datasets.load_iris()

X_train, X_test , Y_train , Y_test = train_test_split(iris['data'],iris['target'],random_state=0)
# To randomise the training data so that each class has equal number of examples
# Parameters : test_size = 0.25 by default , train_size, shuffle = True by default

regr = linear_model.LogisticRegression()	# Logistic Regression Classifier
# regr = svm.SVC()		# Support Vector Classifier
# regr = svm.LinearSVC()		# Linear Support Vector Classifier
# regr =neighbors.KNeighborsClassifier()	# KNN Classifier
#regr = MLPClassifier()				# Neural Network Classifier

regr.fit(X_train,Y_train)

predict = regr.predict(X_test)

print(metrics.accuracy_score(predict,Y_test))

