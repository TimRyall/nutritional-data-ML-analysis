# split data in to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create and train SVM model
reg_param = 1
epsilon = 0.1

model = svm.SVR(kernel='linear', C=reg_param, epsilon=epsilon)
model.fit(X_train, y_train)

# predict the test data
y_hat = model.predict(X_test)

# report the accuracy of the model
mse  = mean_squared_error(y_test, y_hat) # calculate MSE