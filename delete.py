import numpy as np
import pandas as pd
from seaborn import heatmap

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
output_dataset = pd.read_csv('gender_submission.csv')

train_dataset = train_dataset.drop(['Cabin', 'Ticket'], axis = 1)
test_dataset = test_dataset.drop(['Cabin', 'Ticket'], axis = 1)

train_dataset.head()
train_dataset.tail()
train_dataset.info()
len(train_dataset)
train_dataset.describe()
heatmap(test_dataset.isnull(), cbar = False)
train_dataset.isnull().sum()
test_dataset.isnull().sum()

train_dataset["Embarked"].fillna(train_dataset["Embarked"].value_counts().index[0], inplace=True)

print(test_dataset[test_dataset['Fare'].isnull() == True])
test_dataset.at[152, 'Fare'] = '7.8958'


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

title_mapping = {"Dr": 1, "Mr": 2, "Mrs": 3}

combined_dataset = [train_dataset, test_dataset]
for dataset in combined_dataset:
    dataset['Title'] = dataset['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    #dataset['TitleGroup'] = dataset.Title.map(title_dict)
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms','the Countess','Miss','Mme'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Major','Sir','Capt','Col','Don','Jonkheer','Rev','Master','Lady'], 'Mr')
    dataset['Title'] = dataset['Title'].map(title_mapping)

train_dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
    
train_dataset = train_dataset.drop(['Name', 'Title'], axis = 1)
test_dataset = test_dataset.drop(['Name', 'Title'], axis = 1)

print(train_dataset['Title'].value_counts())
print(test_dataset['Title'].value_counts())

combined_dataset = [train_dataset, test_dataset]
temp = 0
for i in train_dataset:
    i['Age'] = 
        temp = temp + 1