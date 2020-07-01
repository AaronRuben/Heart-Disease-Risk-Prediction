#/usr/bin/env/python
################################################
#Georgia Institute of Technology
#Aaron Pfennig
#2020
################################################
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
import pandas as pd
import argparse
import sys
from clean_data import CleanData

def read_data(path_to_data):
    #read data
    return pd.read_csv(path_to_data, sep=',', header=0)

warnings.filterwarnings("ignore", category=FutureWarning)

def linear_regression():
    reg = linear_model.LinearRegression()
    return reg

def bayesian_regression():
    reg = linear_model.BayesianRidge()
    return reg

def logistic_regression():
    reg = linear_model.LogisticRegression()
    return reg

def sgd_classifier():
    clf = linear_model.SGDClassifier()
    return clf

def sgd_regressor():
    reg = linear_model.SGDRegressor()
    return reg

def support_vector_classifier():
    clf = svm.SVC()
    return clf

def support_vector_regression():
    clf = svm.SVR()
    return clf

def nearest_neighbor_classifier():
    clf = neighbors.KNeighborsClassifier(n_neighbors=2)
    return clf

def nearest_neighbor_regression():
    reg = neighbors.KNeighborsRegressor(n_neighbors=2)
    return reg

def gaussian_process_classifier():
    clf = gaussian_process.GaussianProcessClassifier()
    return clf

def gaussian_process_regressor():
    reg = gaussian_process.GaussianProcessRegressor()
    return reg

def gaussian_naive_bayes():
    gnb = naive_bayes.GaussianNB()
    return gnb

def decision_tree_classifier():
    clf = tree.DecisionTreeClassifier()
    return clf

def decision_tree_regressor():
    reg = tree.DecisionTreeRegressor()
    return reg

def random_forest_classifier():
    clf = ensemble.RandomForestClassifier()
    return clf

def random_forest_regressor():
    reg = ensemble.RandomForestRegressor()
    return reg

def extra_tree_classifier():
    clf = ensemble.ExtraTreesClassifier()
    return clf

def extra_tree_regressor():
    reg = ensemble.ExtraTreesRegressor()
    return reg

def adaboost_classifier():
    clf = ensemble.AdaBoostClassifier()
    return clf

def adaboost_regressor():
    reg = ensemble.AdaBoostRegressor()
    return reg

def gradient_boosting_classifier():
    clf = ensemble.GradientBoostingClassifier()
    return clf

def gradient_boosting_regressor():
    reg = ensemble.GradientBoostingRegressor()
    return reg

def voting_classifier():
    clf = ensemble.VotingClassifier()
    return clf

def fit_model(model, X, y):
    model.fit(X, y)
    return model

def test_model(model, X):
    return model.predict(X)

def cross_validate_model(model, X, y):
    return cross_val_score(model, X, y, cv=5)

def evaluate_different_classifiers(X, y):
    # list of classifers
    models = [sgd_classifier(),
             support_vector_classifier(),
             nearest_neighbor_classifier(),
             gaussian_process_classifier(), gaussian_naive_bayes(),
             decision_tree_classifier(),
             random_forest_classifier(),
             extra_tree_classifier(),
             adaboost_classifier(),
             gradient_boosting_classifier(),]
    for model in models:
        # cross validate each classifier and retrieve scores
        scores = cross_validate_model(model, X, y)
        print(f"{type(model)}:\t{scores.mean()} +\- {scores.std()}")
        
def evaluate_different_regressors(X, y):
    # list of regressors
    regressors = [linear_regression(), bayesian_regression(), logistic_regression(),
                  sgd_regressor(),
                  support_vector_regression(),
                  gaussian_process_regressor(),
                  decision_tree_regressor(),
                  random_forest_regressor(),
                  extra_tree_regressor(),
                  adaboost_regressor(),
                  gradient_boosting_regressor()
                 ]
    # initialize CV
    cv = StratifiedKFold(n_splits=5)
    # cross validate each regressor
    for reg in regressors:
        scores = []
        for train, test in cv.split(X, y):
            # train
            reg = reg.fit(X.iloc[train].values, y.iloc[train].values)
            # predict
            pred = np.where(reg.predict(X.iloc[test].values) > 0, 0, 1)
            # evaluate
            score = accuracy_score(y.iloc[test].values, pred)
            if score < 0.5:
                score = 1 - score
            scores.append(score)
        print(f"{type(reg)}:\t{np.mean(scores)} +\- {np.std(scores)}")

def main(argv):
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='csv file with data. First line contains column names and last column class labels')
    args = parser.parse_args()
    data = read_data(args.input)
    # clean data
    if data.isnull().any().any():
        data = CleanData()(data, 6)
    #data.dropna(inplace=True)
    # extract data values and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    evaluate_different_classifiers(X, y)
    evaluate_different_regressors(X, y)

if __name__ == '__main__':
    main(sys.argv[1:])
