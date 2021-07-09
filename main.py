from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np

iris = datasets.load_iris();

# split in to features and labels

x = iris.data
y = iris.target

# split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2) # 20% for testing

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
