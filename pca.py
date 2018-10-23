import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from csv import reader
from pylab import *

###DATA PREPROCESS
with open('iris2.csv') as f:
    raw_data = f.read()

####PREPROCESS OF THE DATASET######
def data_preprocess(raw_data):
    # Load a CSV file
    dataset = list()
    #with filename as file:
    csv_reader = reader(raw_data.split('\n'), delimiter=',')
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)

    pd_data = pd.DataFrame(dataset)

    labels = pd_data.iloc[:,-1].values
    labels = labels[:, np.newaxis]
    #CONVERTING TEXT CLASS LABELS TO NUMBERS
    b, c = np.unique(labels, return_inverse=True)
    labels = c[:, np.newaxis] + 1
    labels = pd.DataFrame(labels)

    pd_data.drop(pd_data.columns[len(pd_data.columns)-1], axis=1, inplace=True)

    result = pd.concat([pd_data, labels], axis=1)
    dataset = result.values
    dataset = np.array(dataset).astype(np.float)


    # Find the min and max values for each column
    stats = [[min(column), max(column)] for column in zip(*dataset)]

    # Rescale dataset columns to the range 0-1 - normalization
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - stats[i][0]) / (stats[i][1] - stats[i][0])
    return dataset

dataset = data_preprocess(raw_data)
df = pd.DataFrame(dataset)
df_x = df.iloc[:,0:-1]
df_y = df.iloc[:,-1]
#print(df_x.describe())
#print(df_y.describe())

###APPLY LINEAR REGRESSION
reg = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
reg.fit(x_train, y_train)
print(reg.score(x_test, y_test))

###APPLY PCA
pca = PCA(n_components=2, whiten='True')
x = pca.fit(df_x).transform(df_x)
x = pd.DataFrame(x)
print(pca.explained_variance_)

###LINEAR REGRESSION ON 2 FEATURES
reg = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, df_y, test_size=0.2, random_state=4)
reg.fit(x_train, y_train)
print(reg.score(x_test, y_test))

###MAKE PLOTS
y = df_y.values.astype(np.int)
plt.style.use('seaborn-white')
fig = plt.figure(figsize=(18,6))

plt.subplot(1, 3, 1)
title('Class 1')
for i in range(0, df.shape[0]):
    if y[i]==1:
        plt.scatter(x.iloc[i, 1], x.iloc[i, 0], c='r')
    else:
        plt.scatter(x.iloc[i, 1], x.iloc[i, 0], c='g')

plt.subplot(1, 3, 2)
title('Class 2')
for i in range(0, df.shape[0]):
    if y[i]==2:
        plt.scatter(x.iloc[i, 1], x.iloc[i, 0], c='r')
    else:
        plt.scatter(x.iloc[i, 1], x.iloc[i, 0], c='g')

plt.subplot(1, 3, 3)
title('Class 3')
for i in range(0, df.shape[0]):
    if y[i]==3:
        plt.scatter(x.iloc[i, 1], x.iloc[i, 0], c='r')
    else:
        plt.scatter(x.iloc[i, 1], x.iloc[i, 0], c='g')
plt.show()