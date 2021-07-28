# Data Preprocessing Template with Python
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Delineate the training and testing datasets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
