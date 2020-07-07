#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Import needed packages
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# load the dataset - NEEDS FIXED TO BE WITH ACTUAL CLEANED DATA
data = pd.read_csv(r"C:\Users\Elianna\Documents\2020 Summer\ML\Project/datasets_222487_478477_framingham-cleanonly.csv")
# Remove label from X and define labels as y
X = data.iloc[:, 0:15]
y = data[data.columns[15]]

# Standardize the data
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X = sc.fit_transform(X)

# VERSION 1 - Implementing and reporting explained variance ratio for each principle component
print("V1")
pca = PCA()
X_PCA = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# VERSION 2 - Implement PCA for all possible number of components, and report accuracy, variance, and make box and whisker plot for each
# Following this tutorial structure : https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/
print("V2")

# get a list of models to evaluate
def get_models():
	models = dict()
	for i in range(1,16):
		steps = [('pca', PCA(n_components=i)), ('m', LogisticRegression())]
		models[str(i)] = Pipeline(steps=steps)
	return models

# evaluate a given model using cross-validation - IS THIS A GOOD TEST FOR US? 
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# define dataset
print("print x and y")
print(X)
print(y)
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
pyplot.show()


# In[ ]:




