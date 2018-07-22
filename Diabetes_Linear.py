import matplotlib.pyplot as plt 
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# print (diabetes)

# For now, use only 1 feature of the dataset
diabetes_X = diabetes.data[:,np.newaxis,2]		# Can use reshape method as well
print(diabetes_X.shape)					# Or to get 2D array, instead used 2:3
#print(diabetes_X)

# Split the data into training and testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets (Labels) into training and testing sets 
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
print (diabetes_y_train.shape)
#print (diabetes_y_test)

# Create Linear Regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# Predict output of training set ( To check accuracy )
#diabetes_y_train_predict = regr.predict(diabetes_X_train)	

#print ("Accuracy : " , accuracy_score(diabetes_y_test,diabetes_y_pred))

# The co-efficients
print(" Co-efficients: ",regr.coef_)
# Mean squared error
print ("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test,diabetes_y_pred))
# Variance score : 1 is perfect prediction
print (" Variance score : %.2f" % r2_score(diabetes_y_test,diabetes_y_pred))


# Plot outputs for testing datapoints
plt.scatter(diabetes_X_test, diabetes_y_test, color = 'black')
plt.plot(diabetes_X_test, diabetes_y_pred, color = 'blue' , linewidth = 3)

# Plot outputs for training datapoints
#plt.scatter(diabetes_X_train, diabetes_y_train, color = 'yellow')
#plt.plot(diabetes_X_train, diabetes_y_train_predict, color = 'orange')

plt.xticks(())
plt.yticks(())


plt.show()
