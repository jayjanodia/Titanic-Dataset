# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 07:35:52 2019

@author: jayja
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier,ExtraTreesClassifier,BaggingClassifier,RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

heatmap(train_set.isnull(), cbar = False)
train_set.isnull().sum()
test_set.isnull().sum()

combined_set = [train_set, test_set]

for dataset in combined_set:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
del dataset

train_set["Embarked"].fillna(train_set["Embarked"].value_counts().index[0], inplace=True)

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
    dataset['FamilySize'] = dataset['SibSp'] + train_set['Parch']
    dataset['HasCabin'] = dataset['Cabin'].notnull().astype(int)
    dataset['TitleGroup'] = dataset.Title.map(title_dict)
    
    
print(train_set['TitleGroup'].value_counts())

print(test_set[test_set['TitleGroup'].isnull() == True])
test_set.at[414, 'TitleGroup'] = 'Noble'


train_set = pd.get_dummies(train_set, columns = ['Sex'], drop_first = True)
#train_set = pd.get_dummies(train_set, columns = ['Pclass'], drop_first = True)
train_set = pd.get_dummies(train_set, columns = ['Embarked'], drop_first = True)
test_set = pd.get_dummies(test_set, columns = ['Sex'], drop_first = True)
#test_set = pd.get_dummies(train_set, columns = ['Pclass'], drop_first = True)
test_set = pd.get_dummies(test_set, columns = ['Embarked'], drop_first = True)

x_train = train_set.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'Title', 'SibSp', 'Parch'], axis = 1)
x_test = test_set.drop(['Name', 'Ticket', 'Cabin', 'Title', 'SibSp', 'Parch'], axis = 1)
titlegroup_labels = list(set(title_dict.values()))
titlegroup_dict = dict(zip(titlegroup_labels, list(range(len(titlegroup_labels)))))
x_train['TitleGroup'] = x_train['TitleGroup'].map(titlegroup_dict).astype(int)
x_test['TitleGroup'] = x_test['TitleGroup'].map(titlegroup_dict).astype(int)
#x_train = pd.get_dummies(x_train, columns = ['TitleGroup'], drop_first = True)
#x_test = pd.get_dummies(x_test, columns = ['TitleGroup'], drop_first = True)
y_train = train_set['Survived']
y_test = pd.read_csv('gender_submission.csv')
y_test = y_test['Survived']
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
    
    
x1 = [x_train, x_test]
y1 = [y_train, y_test]
kfold = KFold(n_splits = 10, random_state = 0)
estimators = []
model1 = LogisticRegression()
#self.model.fit(self.x_train, self.y_train)
model2 = KNeighborsClassifier(n_neighbors= 5, metric = 'minkowski', p =  2)
model3 = SVC(kernel = 'linear', random_state = 0)
model4 = GaussianNB()
model5 = DecisionTreeClassifier(criterion='entropy', random_state = 0)
model6 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
estimators.append(('logreg', model1))
estimators.append(('knn', model2))
estimators.append(('svm', model3))
estimators.append(('nb', model4))
estimators.append(('dec_tree', model5))
estimators.append(('random_tree', model6))
ensemble = VotingClassifier(estimators)
#results = cross_val_score(ensemble, x_train, y_train, cv = kfold)
result1 = model1.fit(x_train, y_train)
result2 = model2.fit(x_train, y_train)
result3 = model3.fit(x_train, y_train)
result4 = model4.fit(x_train, y_train)
result5 = model5.fit(x_train, y_train)
result6 = model6.fit(x_train, y_train)
result6.score(x_test, y_test)
#for i in results:
#    print(f'Voting======\nMax: {i.max()}\nMean: {i.mean()}')
    
model = SVC(kernel = 'linear', random_state = 0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
heatmap(cm, annot = True)
print(f'Accuracy {accuracy_score(y_test, y_pred)*100} and f1 score {f1_score(y_test, y_pred)*100}')


passengerId = np.array(test_set['PassengerId']).astype(int)
submission = pd.DataFrame({ 'PassengerId' : passengerId, 'Survived' : y_pred })
print(submission.shape)

submission.to_csv('submission.csv', index=False)