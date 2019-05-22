# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:49:52 2019

@author: dblankson
"""
#library to import the dataset
import pandas as pd

dataset=pd.read_csv('train.csv')
# Drop features that are not important to the data analysis
dataset=dataset.drop([ 'Name','PassengerId','Name','Ticket','Cabin'],axis=1)
#Drop the 2 missing values in the embarked column
dataset = dataset[pd.notnull(dataset['Embarked'])]
#Splitting the dataset into x and y
y=dataset.iloc[:,0:1].values
x=dataset.iloc[:,1:9].values
# x is an object type. To see x,we convert it to dataframe
df=pd.DataFrame(x)

#checking for total missing values in all the columns
dataset.isnull().sum()
#replacing the age column of missing values with the mean
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,2:3])
x[:,2:3]=imputer.transform(x[:,2:3])
#verifying for no missing values
df.isnull().sum()
#Taking care of categorical data in the embarked and gender columns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
x[:,1]=le.fit_transform(x[:,1])
x[:,6]=le.fit_transform(x[:,6])
onehotencoder=OneHotEncoder(categorical_features=[1])
onehotencoder=OneHotEncoder(categorical_features=[6])
x=onehotencoder.fit_transform(x).toarray()

#avoid the dummy variable trap
x=x[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr=classification_report(y_test,y_pred)

#applying kfold cross validation to test the model on all the training set
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()

#grid search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
        {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7]}]

grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_












