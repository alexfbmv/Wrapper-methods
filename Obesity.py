# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:26:02 2022

@author: Alex
"""

# import libraries
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


df = pd.read_csv("Obesity.csv")


# Encode categorical variables with label encoder
df['Gender']= df['Gender'].astype('category')
df['Gender']= df['Gender'].cat.codes


df['family_history_with_overweight']= df['family_history_with_overweight'].astype('category')
df['family_history_with_overweight']= df['family_history_with_overweight'].cat.codes


df['FAVC']= df['FAVC'].astype('category')
df['FAVC']= df['FAVC'].cat.codes

df['CAEC']= df['CAEC'].astype('category')
df['CAEC']= df['CAEC'].cat.codes

df['SMOKE']= df['SMOKE'].astype('category')
df['SMOKE']= df['SMOKE'].cat.codes

print(df.loc[0:2,'CALC'])

df['SCC']= df['SCC'].astype('category')
df['SCC']= df['SCC'].cat.codes

df['CALC']= df['CALC'].astype('category')
df['CALC']= df['CALC'].cat.codes

df['NObeyesdad']= df['NObeyesdad'].astype('category')
df['NObeyesdad']= df['NObeyesdad'].cat.codes


# Dummy code transportation variable 

ohe= pd.get_dummies(df['MTRANS'])
df = df.join(ohe)

# Split the data into predictor variables and an outcome variable
X = df.drop(['MTRANS'], axis = 1)
X = X.drop(['NObeyesdad'], axis = 1)

y = df['NObeyesdad']

# Create an LR model and .fit() it
lr = LogisticRegression(max_iter=100)
lr.fit(X,y)

# Print the accuracy of the model
print(lr.score(X,y))

# 1.Create a sequential forward selection model
sfs = SFS(lr,
          k_features = 9,
          forward = True, 
          floating = False,
          scoring = 'accuracy',
          cv=0)

# Fit the sequential forward selection model to X and y
sfs.fit(X,y)

# Inspect the results of sequential forward selection
print(sfs.subsets_)

# See which features sequential forward selection chose
print(sfs.subsets_[9]['feature_names'])

# Print the model accuracy after doing sequential forward selection
print(sfs.subsets_[9]['avg_score'])

# Plot the model accuracy as a function of the number of features used
plot_sfs(sfs.get_metric_dict())
plt.show


# 2.Create a sequential backward selection model
sbs = SFS(lr,
          k_features = 7,
          forward = False, 
          floating = False,
          scoring = 'accuracy',
          cv=0)

# Fit the sequential backward selection model to X and y
sbs.fit(X,y)

# Inspect the results of sequential backward selection
print(sbs.subsets_[7])

# See which features sequential backward selection chose
print(sbs.subsets_[7]['feature_names'])

# Print the model accuracy after doing sequential backward selection
print(sbs.subsets_[7]['avg_score'])

# Plot the model accuracy as a function of the number of features used
plot_sfs(sbs.get_metric_dict())
plt.show

# 3. Apply Recursive Feature Elimination Method
# Standardize the data
Xstd = StandardScaler().fit_transform(X)
Xstd = pd.DataFrame(StandardScaler().fit_transform(X))

# Create a recursive feature elimination model
rfe = RFE(lr, n_features_to_select = 8)

# Fit the recursive feature elimination model to X and y
rfe.fit(X,y)

# See which features recursive feature elimination chose
print(rfe.support_)

features = X.columns.tolist()
rfe_features = [f for (f, support) in zip(features, rfe.support_) if support]
print(rfe_features)

# Print the model accuracy after doing recursive feature elimination
print(rfe.score(X, y))
