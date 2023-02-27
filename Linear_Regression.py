import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# from hw7
def normalize_train(X_train):

    # fill in
    X = []
    mean = []
    std = []
    for i in range(len(X_train[0])):
        x = X_train[:,i]
        x_mean = np.mean(x)
        x_std = np.std(x)
        temp = []
        for n in x:
            temp.append((n - x_mean) / x_std)
        X.append(temp)
        mean.append(x_mean)
        std.append(x_std)
    
    
    X = np.transpose(X)
    return X, mean, std

def normalize_test(X_test, trn_mean, trn_std):

    # fill in
    X = []
    for i in range(len(X_test[0])):
        x = X_test[:,i]
        temp = []
        for n in x:
            temp.append((n - trn_mean[i]) / trn_std[i])
        X.append(temp)
    
    X = np.transpose(X)

    return X


def get_lambda_range():
    
    # fill in
    lmbda = np.logspace(-1, 3, 51)
    return lmbda


def train_model(X, y, l):

    # fill in
    reg = Ridge(alpha = l, fit_intercept=True)
    model = reg.fit(X,y)

    return model


def error(X, y, model):

    # Fill in
    y_pred = model.predict(X)
    mse = mean_squared_error(y_pred, y)
    return mse


def regression(df):
    X = np.array(df.drop(["s"], axis=1))
    y = np.array(df["s"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    # Define the range of lambda to test
    lmbda = get_lambda_range()
    #lmbda = [1,3000]
    MODEL = []
    MSE = []
    for l in lmbda:
        # Train the regression model using a regularization parameter of l
        model = train_model(X_train, y_train, l)

        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)

        # Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)


    ind = MSE.index(np.min(MSE))# fill in
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]

    print(
        "Best lambda tested is "
        + str(lmda_best)
        + ", which yields an MSE of "
        + str(MSE_best)
    )

    return model_best

    