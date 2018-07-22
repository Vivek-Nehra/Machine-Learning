import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import accuracy_score

# Import some data 
iris = datasets.load_iris()
X = iris.data[:,:2]	# We only take the first two features
Y = iris.target

h = .02	# Step size in the mesh

# Creating a logistic regresiion object
logreg = linear_model.LogisticRegression(C=1e5)

# We create an instance of Neighbours Classifier and fit the data
logreg.fit(X,Y)

# Plot the decision boundary. For that , we will assign a color to each
# point in the mesh [ x_min , x_max] x [y_min, y_max].
x_min,x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
y_min,y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
xx,yy = np.meshgrid(np.arange(x_min,x_max,h)
Z=logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize = (4,3))
plt.pcolormesh(xx,yy,Z, cmap = plt.cm.Paired)

# PLot also the training points
plt.scatter(X[:,0],X[:,1] , c=Y, edgecolors = 'k' , cmap= plt.cm.Paired)
plt.xlabel('Sepal length')
plt. ylabel('Sepal width')

plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xticks(())
plt.yticks(())

