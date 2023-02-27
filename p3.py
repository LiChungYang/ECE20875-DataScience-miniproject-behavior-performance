import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def getdata():
    dataset_1 = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    dataset_1['Total'] = pd.to_numeric(dataset_1['Total'].replace(',', '', regex=True))
    dataset_1['Precipitation'] = pd.to_numeric(dataset_1['Precipitation'].replace(',', '', regex=True))
    return dataset_1

precip = list(getdata()["Precipitation"])
precip = [float(i) for i in precip]
total = list(getdata()["Total"])
total = [float(i) for i in total]

def normalize_train(X_train):
    mean = np.mean(X_train)
    std = np.std(X_train)
    train = (X_train - mean) / std
    return train, mean, std

def normalize_test(X_test, trn_mean, trn_std):
    x = (X_test - trn_mean) / trn_std
    return x

def train_model(X_train, y_train, l):
    reg = Ridge(alpha = l, fit_intercept = True)
    model = reg.fit(X_train, y_train)
    return model

def error(X_test, y_test, model):
    y_pre = model.predict(X_test)
    mse = mean_squared_error(y_pre, y_test)
    return mse

def linreg_one(stat, total):
    X = np.array([stat]).T
    y = np.array([total]).T
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)
    model = train_model(X_train, y_train, 1)
    model.fit(X_train, y_train)
    s = model.score(X_test, y_test)
    mse = error(X_test, y_test, model)
    return s, mse

X = total
y = np.array([precip])
y = [1 if i > 0 else 0 for i in precip]
s, mse = linreg_one(X,y)
print("The coefficient of correlation between precipitation and traffic:", s)
print("The mean square error of precipitation and traffic:", mse)