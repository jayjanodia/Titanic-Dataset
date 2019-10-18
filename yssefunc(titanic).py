# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 07:26:14 2019

@author: jayja
"""
#importing the library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

#import dataset
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
train_set.head()
train_set.tail()
train_set.info()
len(train_set)
train_set.describe()
heatmap(train_set.isnull(), cbar = False)
train_set.isnull().sum()
test_set.isnull().sum()

combined_set = [train_set, test_set]

for dataset in combined_set:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
del dataset
    
train_set["Embarked"].fillna(train_set["Embarked"].value_counts().index[0], inplace=True)

age_bins = [0,15,35,45,60,200]
age_labels = ['15-','15-35','35-45','40-60','60+']
fare_bins = [0,10,30,60,999999]
fare_labels = ['10-','10-30','30-60','60+']

title_dict = {
                "Mr" :        "Mr",
                "Miss" :      "Miss",
                "Mrs" :       "Mrs",
                "Master" :    "Master",
                "Dr":         "Scholar",
                "Rev":        "Religious",
                "Col":        "Officer",
                "Major":      "Officer",
                "Mlle":       "Miss",
                "Capt":       "Noble",
                "Ms":         "Noble",
                "Mme":        "Mrs",
                "Lady":       "Mrs",
                "Sir":        "Noble",
                "Don" :       "Noble",
                "Jonkheer":   "Noble",
                "the Countess":"Noble"
            }

for dataset in combined_set:
    dataset['Title'] = dataset['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    dataset['AgeRange'] = pd.cut(dataset['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
    dataset['FareRange'] = pd.cut(dataset['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)
    dataset['FamilySize'] = dataset['SibSp'] + train_set['Parch']
    dataset['HasCabin'] = dataset['Cabin'].notnull().astype(int)
    dataset['Family'] = ''
    dataset.loc[dataset['FamilySize'] == 0, 'Family'] = 'alone'
    dataset.loc[(dataset['FamilySize'] > 0) & (dataset['FamilySize'] <= 3), 'Family'] = 'small'
    dataset.loc[(dataset['FamilySize'] > 3) & (dataset['FamilySize'] <= 6), 'Family'] = 'medium'
    dataset.loc[dataset['FamilySize'] > 6, 'Family'] = 'large'
    dataset['TitleGroup'] = dataset.Title.map(title_dict)
    
del age_bins, fare_bins

print(train_set['Title'].value_counts())

print(test_set[test_set['TitleGroup'].isnull() == True])

test_set.at[414, 'TitleGroup'] = 'Noble'

train_set = pd.get_dummies(train_set, columns = ['Sex'], drop_first = True)
#train_set = pd.get_dummies(train_set, columns = ['Pclass'], drop_first = True)
train_set = pd.get_dummies(train_set, columns = ['Embarked'], drop_first = True)
train_set = pd.get_dummies(train_set, columns = ['Family'], drop_first = True)
test_set = pd.get_dummies(test_set, columns = ['Sex'], drop_first = True)
#test_set = pd.get_dummies(train_set, columns = ['Pclass'], drop_first = True)
test_set = pd.get_dummies(test_set, columns = ['Embarked'], drop_first = True)
test_set = pd.get_dummies(test_set, columns = ['Family'], drop_first = True)

#x_train = train_set.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'Title', 'FamilySize'], axis = 1)
x_train = train_set.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'Title', 'FamilySize', 'SibSp', 'Parch', 'AgeRange', 'FareRange'], axis = 1)
#x_test = test_set.drop(['Name', 'Ticket', 'Cabin', 'Title', 'FamilySize'], axis = 1)
x_test = test_set.drop(['Name', 'Ticket', 'Cabin', 'Title', 'FamilySize', 'SibSp', 'Parch', 'AgeRange', 'FareRange'], axis = 1)
#agerange_dict = dict(zip(age_labels, list(range(len(age_labels)))))
#x_train['AgeRange'] = x_train['AgeRange'].map(agerange_dict).astype(int)
#x_test['AgeRange'] = x_test['AgeRange'].map(agerange_dict).astype(int)
#farerange_dict = dict(zip(fare_labels, list(range(len(fare_labels)))))
#x_train['FareRange'] = x_train['FareRange'].map(farerange_dict).astype(int)
#x_test['FareRange'] = x_test['FareRange'].map(farerange_dict).astype(int)
titlegroup_labels = list(set(title_dict.values()))
titlegroup_dict = dict(zip(titlegroup_labels, list(range(len(titlegroup_labels)))))
x_train['TitleGroup'] = x_train['TitleGroup'].map(titlegroup_dict).astype(int)
x_test['TitleGroup'] = x_test['TitleGroup'].map(titlegroup_dict).astype(int)
y_train = train_set['Survived']
model = XGBClassifier()
model.fit(x_train, y_train)
print(model.feature_importances_)
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plot_importance(model)
plt.show()

x1_train, x1_test, y1_train, y1_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
model = XGBClassifier()
model.fit(x1_train, y1_train)
y_pred = model.predict(x1_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y1_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_x_train = selection.transform(x1_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_x_train, y1_train)
	# eval model
	select_X_test = selection.transform(x1_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y1_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_x_train.shape[1], accuracy*100.0))
    
    



