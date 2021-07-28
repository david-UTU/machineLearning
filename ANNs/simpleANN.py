from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Data Preprocessing

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data through 'Gender'
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
#
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize the ANN
ann = tf.keras.models.Sequential()

# Input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# ANN compiling
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training phase
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# Prediction of a single value

print(ann.predict(sc.transform(
    [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# Test set predictions
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# CONFUSION MATRIX TIME
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
