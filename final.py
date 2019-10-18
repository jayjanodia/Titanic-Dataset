# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:40:29 2019

@author: jayja
"""

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import re


from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB





#importing the dataset
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
sns.heatmap(train_set.isnull())
train_set.isnull().sum().sort_values(ascending=False)
test_set.isnull().sum()

train_set.info()
test_set.info()
train_set.describe()
test_set.describe()
train_set.columns.values


#Correlation between embarked, pclass and sex. Execute all 3 lines together.
FacetGrid1 = sns.FacetGrid(train_set, row = 'Embarked', height = 4.5, aspect = 1.6) #creates empty graphs of height 4.5 and aspect ratio 1:6
FacetGrid1.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None)
FacetGrid1.add_legend()

#Correlation between Pclass and survived
#%matplotlib qt
sns.barplot(x='Pclass', y='Survived', data=train_set)
#Creating a new column to show respective person's title
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

titles = {
                "Mr" :        1,
                "Miss" :      2,
                "Mrs" :       3,
                "Master" :    4,
                "Scholar":    5,
                "Religious":  6,
                "Officer":    7,
                "Noble":      8,
        }

#Creating a new column to show respective person's age in case that person's age was nan
age_dict = {
                "Mr" :        30,
                "Miss" :      14,
                "Mrs" :       30,
                "Master" :    14,
                "Dr":         35,
                "Rev":        46,
                "Col":        58,
                "Major":      47,
                "Mlle":       24,
                "Capt":       70,
                "Ms":         28,
                "Mme":        24,
                "Lady":       48,
                "Sir":        49,
                "Don" :       40,
                "Jonkheer":   38,
                "the Countess":33
            }

#Creating a new column family size to show how many siblings and parents and children a member has
combined_set = [train_set, test_set]
for dataset in combined_set:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
    dataset['Title'] = dataset['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    dataset['TitleGroup'] = dataset.Title.map(title_dict)
    dataset['AgeTitleWise'] = dataset.Title.map(age_dict)
    dataset['Age'].fillna(dataset['AgeTitleWise'], inplace = True)
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


sns.barplot(x='Title', y = 'Age', data = train_set)

#On seeing whether null value exists in train_set and test_set, we find that titlegroup for test_set has 1 null value
print(test_set[test_set['TitleGroup'].isnull() == True]) #we see the null value is in index 414 cause that person does not have a title
test_set.at[414, 'TitleGroup'] = 'Noble' #we will assume that they are noble

for dataset in combined_set:
    dataset['TitleGroup'] = dataset['TitleGroup'].map(titles)
#train_set['Age'] = np.where(train_set.Title)


train_set["Embarked"].fillna(train_set["Embarked"].value_counts().index[0], inplace=True)

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
combined_set = [train_set, test_set]

for dataset in combined_set:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature
train_set = train_set.drop(['PassengerId','Cabin', 'Ticket', 'AgeTitleWise', 'Name', 'Title'], axis=1)
test_set = test_set.drop(['PassengerId','Cabin', 'Ticket', 'AgeTitleWise', 'Name', 'Title'], axis=1)


train_set = pd.get_dummies(train_set, columns = ['Sex'], drop_first = True)
test_set = pd.get_dummies(test_set, columns = ['Sex'], drop_first = True)
train_set = pd.get_dummies(train_set, columns = ['Embarked'], drop_first = True)
test_set = pd.get_dummies(test_set, columns = ['Embarked'], drop_first = True)


#use sklearn's qcut() function to see how to divide your column
pd.qcut(train_set['Age'], 7)

data = [train_set, test_set]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 22), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 28), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 30), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 43), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 43) & (dataset['Age'] <= 80), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 80, 'Age'] = 6

# let's see how it's distributed 
train_set['Age'].value_counts()

pd.qcut(train_set['Fare'], 6)


for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train_set['Fare'].value_counts()

train_set.head(10)


for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['FamilySize']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
    
    
    
#Building the model
    
x_train = train_set.drop('Survived', axis = 1)
y_train = train_set['Survived']

x_test = test_set

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
sgd.score(x_train, y_train)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)

knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(x_train, y_train)  
y_pred = knn.predict(x_test)  
acc_knn = round(knn.score(x_train, y_train) * 100, 2)

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

perceptron = Perceptron(max_iter=5)
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)  
y_pred = decision_tree.predict(x_test)  
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, x_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

predictions = cross_val_predict(random_forest, x_train, y_train, cv=3)
confusion_matrix(y_train, predictions)

f1_score(y_train, predictions)


y_test = pd.read_csv('gender_submission.csv')
y_test = y_test.drop(['PassengerId'], axis = 1)

accuracy = accuracy_score(y_test, y_pred)


test_set_dec_tree = pd.read_csv('test.csv')
passengerId = np.array(test_set_dec_tree['PassengerId']).astype(int)
submission = pd.DataFrame({ 'PassengerId' : passengerId, 'Survived' : y_pred })
print(submission.shape)

submission.to_csv('submission_dec_tree.csv', index=False)