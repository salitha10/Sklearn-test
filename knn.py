import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')

x = data[['buying',
          'maint',
          'safety']].values

y = data[['class']]

print(x)

# Converting data
Le = LabelEncoder()
for i in range(len(x[0])):
    x[:, i] = Le.fit_transform(x[:, i])

print(x)

# y data converting
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y)
print(y)

# create model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
knn.fit()
