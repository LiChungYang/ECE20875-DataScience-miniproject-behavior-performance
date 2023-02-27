import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_1 = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pd.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pd.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pd.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pd.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pd.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Total'] = pd.to_numeric(dataset_1['Total'].replace(',', '', regex=True))
dataset_1['High Temp'] = pd.to_numeric(dataset_1['High Temp'].replace(',','', regex=True))
dataset_1['Low Temp'] = pd.to_numeric(dataset_1['Low Temp'].replace(',','', regex=True))
dataset_1['Precipitation'] = pd.to_numeric(dataset_1['Precipitation'].replace(',','', regex=True))

print(dataset_1.describe(include=[np.number]))
plt.subplot(3, 1, 1)
plt.hist(dataset_1['Precipitation'], bins = 10)
plt.xlabel('Precipitation (inch)')
plt.ylabel('Frequency')
plt.subplot(3, 1, 2)
plt.hist(dataset_1['High Temp'], bins = 10)
plt.xlabel('Highest Temp in one day (F)')
plt.ylabel('Frequency')
plt.subplot(3, 1, 3)
plt.hist(dataset_1['Low Temp'], bins = 10)
plt.xlabel('Lowest Temp in one day (F)')
plt.ylabel('Frequency')
plt.show()