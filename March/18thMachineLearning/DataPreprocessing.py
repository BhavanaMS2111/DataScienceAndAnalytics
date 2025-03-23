# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 19:40:28 2025

@author: Hithaardh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'D:\Data Science and Machine Learning\March\18thMachineLearning\Data.csv')

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:,0])
X[:,0] = labelencoder_X.fit_transform(X[:,0])


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



from sklearn.model_selection import train_test_split
#X_Train,X_Test,y_Train,y_test = train_test_split(X,y,test_size=.3,train_size=.8,random_state=0)
X_Train,X_Test,y_Train,y_test = train_test_split(X,y,test_size=.2,random_state=0)


