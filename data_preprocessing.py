# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 07:47:38 2019

@author: jayja
"""

import pandas as pd
import numpy as np

#IMPORTING THE DATASET
dataset = pd.read_csv('train.csv')
#dataset.isna().sum() #prints where nan values are in the dataset
required_columns = ['Pclass', 'Sex', 'Age']
x = dataset[required_columns]
req_col = ['Survived']
y = dataset[req_col]
#y = dataset.iloc[:, 2].values

#REPLACING MISSING DATA BY MEAN
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(x.iloc[:, 2:3])
x.iloc[:, 2:3] = imputer.transform(x.iloc[:, 2:3])
#np.set_printoptions(threshold = np.nan)

#CATEGORIZING TEXT DATA INTO NUMBERS
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x.iloc[:, 1] = labelencoder_x.fit_transform(x.iloc[:, 1])
#INSTEAD OF NUMBERS OF 1 2 3, EACH UNIQUE TEXT GETS IT'S OWN NEW COLUMN
#onehotencoder = OneHotEncoder(categorical_features = [0])
#x = OneHotEncoder.fit_transform(x).toarray()

x = pd.get_dummies(x, prefix=["Age"], drop_first=True)
print(x)