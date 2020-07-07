#!/usr/bin/env python
<<<<<<< HEAD
=======
import pandas as pd
import sys
import argparse
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
from scipy import interp
import joblib
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def perform_gridsearch(X, y, categorical, threads):
    kneighbors = np.arange(4, 11, 2)
    degrees = np.arange(1, 6, 2)
    n_components = np.arange(4, 16, 3)
    n_splits = 5
    methods = ['RF', 'LR', 'SVC']
    #categorical = 'education male currentSmoker prevalentStroke prevalentHyp diabetes'.split(' ')
    best_auc = 0.0
    best_acc = 0.0
    # Split data in train and test set   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nr_combinations = kneighbors.shape[0] * degrees.shape[0] * n_components.shape[0] * len(methods)
    with tqdm(total=nr_combinations) as pbar:
        for method in methods:
            for d in degrees:
                for k in kneighbors:
                    for n in n_components:
                        # prepare training data
                        preprocessing_train = Preprocessing(X_train, y_train, k, d, categorical)
                        preprocessed_data_train = preprocessing_train()
                        dim_reduction_train = DimensionalityReduction(preprocessed_data_train, n)
                        reduced_data_train = dim_reduction_train()
    
                        #prepare test data
                        preprocessing_test = Preprocessing(X_test, y_test, k, d, categorical)
                        preprocessed_data_test = preprocessing_test()
                        dim_reduction_test = DimensionalityReduction(preprocessed_data_test, n)
                        reduced_data_test = dim_reduction_test()
                        
                        classification = Classification(reduced_data_train, y_train, method, n_splits, threads)
                        model, statistics = classification()
                        accuracy = classification.evaluate_model(reduced_data_test, y_test, model)
                        # keep track of best model
                        if accuracy > best_acc:
                            best_acc = accuracy
                            params_best_acc = [method, k, d, n]
                            best_model = model
                            best_statistics = statistics
                        if statistics[-1] > best_auc:
                            best_auc = statistics[-1]
                            params_best_auc = [method, k, d, n]
                        pbar.update(1)
    print(f'Params best ACC:\nMethod: {params_best_acc[0]}\nk: {params_best_acc[1]}\ndegree: {params_best_acc[2]}\nn_components: {params_best_acc[3]}\nACC: {best_acc}')
    print(f'Params best AUC:\nMethod: {params_best_auc[0]}\nk: {params_best_auc[1]}\ndegree: {params_best_auc[2]}\nn_components: {params_best_auc[3]}\nAUC: {best_auc}')
    return best_model, best_statistics, best_acc

class Preprocessing:
    def __init__(self, X, y, k, degree=None, categorical=None):
        """
        Init class
        data: pandas DataFrame, last column represents class labels
        k: int, k-NearestNeighbor are used to impute missing values
        degree: int, do feature engineering --> generate feature with degree less
                     than or equal to specified degree
        """
        if verbose:
            print('Preprocess data')
        self.categorical = categorical
        self.X = X
        self.y = y
        self.k = k
        self.degree = degree
        
    def clean_data(self):
        """
        Imputes missing values using k nearest neighbors
        return: nxd array, cleaned data
        """
        knn = KNNImputer(n_neighbors=self.k)
        cleaned_data = knn.fit_transform(self.X)
        cleaned_data = pd.DataFrame(cleaned_data, columns=self.X.columns)
        # round numbers thus we can get the correct dummies
        categorical_data = cleaned_data[self.categorical].astype(int)
        cleaned_data.drop(self.categorical, inplace=True, axis=1)
        cleaned_data = pd.concat([cleaned_data, categorical_data], axis=1)
        return cleaned_data
    
    def feature_engineering(self, data):
        """
        Feature engineering, generate polynomial feature with degree less than or equal
        to degree specified.
        return: engineered feature
        """
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        data_new = poly.fit_transform(data)
        feature_names = poly.get_feature_names(data.columns)
        return data_new, feature_names

    def check_one_hot_encoding(self, df):
        """
        Perform one hot coding on non-binary, categorical features
        df: pandas DataFrame, nxd with cleaned data
        return: pandas DataFrame with one-hot-encoded categorical features
        """
        for cat in self.categorical:
            # check if binary
            if df[cat].unique().shape[0] == 2:
                continue
            # if not one hot encode
            else:
                df = pd.get_dummies(df, columns=[cat])
        return df

    def standardize_data(self, df):
        """
        By Elianna Paljug
        Standardizes data with mean and std
        df: pd DataFrame nxd
        return: normalized pandas DataFrame nxd
        """
        sc = StandardScaler()
        df = sc.fit_transform(df)
        return df
    
    def __call__(self):
        """
        Clean data and perform feature engineering if degree is provided
        return: pd DataFrame nxd
        """
        # impute missing values
        cleaned_data = self.clean_data()
        cleaned_data = pd.DataFrame(cleaned_data, columns=self.X.columns)
        # one hot encode categorical features
        if not self.categorical is None:
            cleaned_data = self.check_one_hot_encoding(cleaned_data)
        # standardize data
        preprocessed = self.standardize_data(cleaned_data)
        preprocessed = pd.DataFrame(preprocessed, columns=cleaned_data.columns)
        # do feature engineering
        if not self.degree is None:
            preprocessed, feature_names = self.feature_engineering(preprocessed)
            preprocessed = pd.DataFrame(preprocessed, columns=feature_names)
        return preprocessed

class DimensionalityReduction:
    def __init__(self, data, n_components):
        if verbose:
            print('Perform dimensionality reduction')
        self.data = data
        self.n_components = n_components
    
    def pca(self):
        """
        By Elianna Paljug
        Perform PCA
        return: reduced data with dimensions N x n_components
        """
        pca = PCA(n_components=self.n_components)
        reduced = pca.fit_transform(self.data.values)
        columns = [f'PC{i}' for i in range(1, self.n_components + 1)]
        reduced = pd.DataFrame(reduced, columns=columns)
        return reduced

    def __call__(self):
        return self.pca()

class Classification:
    def __init__(self, X, y, method, n_splits, threads):
        if verbose:
            print('Validate and train model')
        self.method = method
        self.n_splits = n_splits
        self.X = X
        self.y = y
        self.threads = threads

    def svc(self):
        """
        Initialize SVC
        """
        svc = SVC(class_weight='balanced', random_state=42, probability=True)
        return svc
    
    def logistic_regression(self):
        """
        Initialize logistic regressor
        """
        lg = LogisticRegression(class_weight='balanced', max_iter=1000,
                                random_state=42, n_jobs=self.threads)
        return lg

    def random_forest(self):
        """
        Initialize Ranfom Forest
        """
        rfc = RandomForestClassifier(class_weight='balanced',
                                     random_state=42, n_jobs=self.threads)
        return rfc

    def neural_net(self):
        raise NotImplementedError

    def perform_cv(self, model):
        """
        Evaluate model in cross validation
        model: model
        return: (mean_fpr, mean_tpr, mean_auc)
        """
        # initialize CV
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for train, test in skf.split(self.X, self.y):
            # train model
            model.fit(self.X.values[train, :], self.y.values[train])
            # predict
            probas_ = model.predict_proba(self.X.values[test, :])
            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(self.y.values[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
        mean_tpr /= skf.get_n_splits(self.X, self.y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        if verbose:
            print(f'Mean AUC: {mean_auc}')
        return (mean_fpr, mean_tpr, mean_auc)

    def train_model(self, model):
        """
        Train model on entire training set
        return: trained model
        """
        model.fit(self.X.values, self.y.values)
        return model

    def predict(self, X, model):
        """
        Predict class labels of X
        X: test data
        model: trained model
        return: predicted class labels
        """
        return model.predict(X)

    def evaluate_model(self, X, y, model):
        """
        Determine accuracy of final model
        """
        ypred = self.predict(X, model)
        acc = accuracy_score(y, ypred)
        if verbose:
            print(f'ACC: {acc}')
        return acc
    
    def __call__(self):
        # initialize model
        if self.method == "SVC":
            model = self.svc()
        elif self.method == 'LR':
            model = self.logistic_regression()
        elif self.method == 'RF':
            model = self.random_forest()
        elif self.method == 'NN':
            model = self.neural_net()
        else:
            raise ValueError("Select one of SVC, LR, RF and NN")
        # cross validate model
        statistics = self.perform_cv(model)
        # train model on entire training set
        model = self.train_model(model)
        return model, statistics

class Visualization:
    def __init__(self, plot_dir):
        self.plot_dir = plot_dir

    def plot_roc_curve(self, fpr, tpr, auc):
        """
        Plot ROC curve based on CV results
        fpr: float, mean false positive rate
        tpr: float, mean true positive rate
        auc: float, mean area under curve
        return: None
        """
        fig, ax = plt.subplots()
        # plot fpr vs tpr
        ax.plot(fpr, tpr, label='ROC AUC: {:.2f}'.format(auc))
        ax.plot([0, 1], [0, 1], linestyle='--', color='red')
        # adjust plot settings
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        # add labels
        ax.set_xlabel('1 - Specificity')
        ax.set_ylabel('Sensitivity')
        ax.legend(loc='lower right')
        figname = self.plot_dir + 'roc_curve.png'
        fig.savefig(figname, bbox_inches='tight')
        
        
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Path to data file in .csv format. Column names should be'\
                        'in line 0 and seperator should be ,. Last column contains labels')
    parser.add_argument('-k', type=int, help='k-Nearest-Neighbor are used to impute missing values, default=8', 
                        default=8)
    parser.add_argument('--degree', type=int, help='Generate polynomial features with degree less or equal to specified degree,'\
                        'default=None', default=None)
    parser.add_argument('--n_components', type=int, help='Number of Components used for PCA, default=7', default=7)
    parser.add_argument('--categorical', nargs='+', help='List of categorical features, separated by a space. They will be one-hot-encoded', required=False, default=None)
    parser.add_argument('--n_splits', type=int, help='Number of splits performed during CV, default=10', default=10)
    parser.add_argument('--method', help='Which supervised learning method to use. One of: SVC, LR (LogisticRegression), RF and NN, default=RF', default='RF')
    parser.add_argument('--verbose', help='Verbosity', default=False, action='store_true')
    parser.add_argument('--optimize', help='Perform GridSearch and print optimal settings', default=False, action='store_true')
    parser.add_argument('--output_dir', help='Directory where to save model etc.', default='./output/')
    parser.add_argument('--threads', help='Number of threads to use when possible, default=8', default=8, type=int)
    args = parser.parse_args()
    global verbose
    verbose = args.verbose
    # parse data
    data_df = pd.read_csv(args.data, sep=',', header=0)
    X = data_df.iloc[:, :-1]
    y = data_df.iloc[:, -1]
    
    # do grid search
    if args.optimize:
        model, statistics, accuracy = perform_gridsearch(X, y, args.categorical, args.threads)
    # run in normal mode
    else:
        # Split data in train and test set   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
        # prepare training data
        preprocessing_train = Preprocessing(X_train, y_train, args.k, args.degree, args.categorical)
        preprocessed_data_train = preprocessing_train()
        dim_reduction_train = DimensionalityReduction(preprocessed_data_train, args.n_components)
        reduced_data_train = dim_reduction_train()
    
        #prepare test data
        preprocessing_test = Preprocessing(X_test, y_test, args.k, args.degree, args.categorical)
        preprocessed_data_test = preprocessing_test()
        dim_reduction_test = DimensionalityReduction(preprocessed_data_test, args.n_components)
        reduced_data_test = dim_reduction_test()

        classification = Classification(reduced_data_train, y_train, args.method, args.n_splits, args.threads)
        model, statistics = classification()
        accuracy = classification.evaluate_model(reduced_data_test, y_test, model)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    # plot roc
    visualize = Visualization(args.output_dir)
    visualize.plot_roc_curve(*statistics)
    # save model
    joblib.dump(model, f'{args.output_dir}trained_model.sav')

if __name__ == '__main__':
    main(sys.argv[1:])
>>>>>>> d91af72... Fixed bug in grid search and updated default parameters
