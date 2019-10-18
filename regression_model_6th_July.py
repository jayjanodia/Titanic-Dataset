# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 16:39:50 2019

@author: jayja
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from seaborn import heatmap
from warnings import filterwarnings
filterwarnings("ignore")
np.random.seed(123)

class Regression(object):
    def __init__(self, filename):
        try:
            self.dataset = pd.read_csv(filename)
            self.dataset = self.dataset.fillna(self.dataset.mean())
            #self.dataset = self.dataset.dropna()
            self.dataset = pd.get_dummies(self.dataset, columns=["Sex"], drop_first=None)
            self.column_names = ["Sex_male", "Pclass", "Age"]
            self.x = self.dataset[self.column_names]
            self.y = self.dataset["Survived"]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=0)
            #print(self.x_train.size/3, self.y_train.size, self.x_test.size/3, self.y_test.size)
            self.model = LogisticRegression()
        except Exception as e:  
            print(e)
        
    def fit(self):
        try:
            self.model.fit(self.x_train, self.y_train)
        except Exception as e:
            print(e)
            return False
        return True
        
    def transform(self):
        try:
            y_pred = self.model.predict(self.x_test)
            cm = confusion_matrix(self.y_test, y_pred)
            heatmap(cm, annot=True)
            print(f'Accuracy {accuracy_score(self.y_test, y_pred)*100} and f1 score {f1_score(self.y_test, y_pred)*100}')
        except Exception as e:
            print(e)
            return None
        return y_pred, self.y_test
    
if __name__ == "__main__":
    reg_model = Regression("train.csv")
    reg_model.fit()
    y_pred = reg_model.transform()