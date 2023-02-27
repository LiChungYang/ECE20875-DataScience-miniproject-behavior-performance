import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def getdata():
    dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
    dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
    dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
    dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_1['Total'] = pandas.to_numeric(dataset_1['Total'].replace(',', '', regex=True))
    return dataset_1


brook = list(getdata()["Brooklyn Bridge"])
manh = list(getdata()["Manhattan Bridge"])
william = list(getdata()["Williamsburg Bridge"])
queen = list(getdata()["Queensboro Bridge"])
total = list(getdata()["Total"])
X = np.array([brook, manh, william, queen]).T
y = np.array(total).T

# problem 1
def normalize_train(X_train):
    mean = np.mean(X_train)
    std = np.std(X_train)
    train = (X_train - mean) / std
    return train, mean, std

def normalize_test(X_test, trn_mean, trn_std):
    x = (X_test - trn_mean) / trn_std
    return x

def get_lambda_range():
    lmbda = np.logspace(-1, 3, num = 51)
    return lmbda

def train_modeal(X_train, y_train, l):
    reg = Ridge(alpha = l, fit_intercept = True)
    model = reg.fit(X_train, y_train)
    return model

def error(X_test, y_test, model):
    y_pre = model.predict(X_test)
    mse = mean_squared_error(y_pre, y_test)
    return mse

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
[X_train, trn_mean, trn_std] = normalize_train(X_train)
X_test = normalize_test(X_test, trn_mean, trn_std)
lmbda = get_lambda_range()

MODEL = []
MSE = []

for i in lmbda:
    model = train_modeal(X_train, y_train, i)
    MODEL.append(model)
    mse = error(X_test, y_test, model)
    MSE.append(mse)
    
i = np.argmin(MSE)
[lmda_best, MSE_best, model_best] = [lmbda[i], MSE[i], MODEL[i]]
print(
        "Best lambda tested is "
        + str(lmda_best)
        + ", which yields an MSE of "
        + str(MSE_best)
    )
print(f"{model_best.coef_[0]} x1(Brooklyn) \n{model_best.coef_[1]} x2(Manhattan) \n{model_best.coef_[2]} x3(Williamsburg) \n{model_best.coef_[3]} x4(Queensboro)")

X_test = normalize_test(X, trn_mean, trn_std)
y_hat = model.predict(X_test)
plt.figure()
plt.title('y_hat vs. y')
plt.plot(y_hat, label = 'y_hat')
plt.plot(y,)
plt.legend(['y_hat','y'])
plt.show()