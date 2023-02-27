import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def getdata():
    dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    dataset_1['High Temp'] = pandas.to_numeric(dataset_1['High Temp'].replace(',','', regex=True))
    dataset_1['Low Temp'] = pandas.to_numeric(dataset_1['Low Temp'].replace(',','', regex=True))
    dataset_1['Precipitation'] = pandas.to_numeric(dataset_1['Precipitation'].replace(',','', regex=True))
    dataset_1['Total'] = pandas.to_numeric(dataset_1['Total'].replace(',', '', regex=True))
    return dataset_1

precip = list(getdata()["Precipitation"])
precip = [float(i) for i in precip]
high_t = list(getdata()["High Temp"])
high_t = [float(i) for i in high_t]
low_t = list(getdata()["Low Temp"])
low_t = [float(i) for i in low_t]
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

def linreg_t_w(high_t, low_t, precip, total):
    X = np.array([high_t, low_t, precip]).T
    y = np.array([total]).T
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)
    model = train_model(X_train, y_train, 1)
    model.fit(X_train, y_train)
    s = model.score(X_test, y_test)
    return s

def linreg_one(stat, total):
    X = np.array([stat]).T
    y = np.array([total]).T
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)
    model = train_model(X_train, y_train, 1)
    model.fit(X_train, y_train)
    s = model.score(X_test, y_test)
    return s

def linreg_two(i, j, total):
    X = np.array([i,j]).T
    y = np.array([total]).T
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)
    model = train_model(X_train, y_train, 1)
    model.fit(X_train, y_train)
    s = model.score(X_test, y_test)
    return s

coef_t_w = linreg_t_w(high_t, low_t, precip, total)
coef_low_t = linreg_one(low_t, total)
coef_high_t = linreg_one(high_t, total)
coef_precip = linreg_one(precip, total)
coef_high_low_t = linreg_two(high_t, low_t, total)
coef_low_precip = linreg_two(low_t, precip, total)
coef_high_precip = linreg_two(high_t, precip, total)

print("The coefficient of correlation between low temperature, high temperature, precipitation and traffic:", coef_low_t)
print("The coefficient of correlation between low temperature and traffic:", coef_low_t)
print("The coefficient of correlation between high temperature and traffic:", coef_high_t)
print("The coefficient of correlation between precipitation and traffic:", coef_precip)
print("The coefficient of correlation between high, low temperature and traffic:", coef_high_low_t)
print("The coefficient of correlation between low temperature, precipitation and traffic:", coef_low_precip)
print("The coefficient of correlation between high temperature, precipitation and traffic:", coef_high_precip)
print("\nFrom above coefficient values, all values are below 0.5 which indicates that there is no strong correlation between weather and traffic.")