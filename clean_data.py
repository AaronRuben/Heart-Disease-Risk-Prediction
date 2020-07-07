#!/usr/bin/env python
################################################
#Georgia Institute of Technology
#Aaron Pfennig
#2020
################################################
import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from sklearn import impute# import KNNImputer

def read_data(path_to_data):
    #read data
    return pd.read_csv(path_to_data, sep=',', header=0)

class CleanData:
    def __init__(self):
        pass
    
    def __call__(self, data, k):
        """
        Initialize visualization class.
        data: pandas DataFrame, contains data, samples in rows features in columns, last column contains class labels
        k: int, number of nearest neighbors to consider to impute missing values
        return: pandas DataFrame, with cleaned data
        """
        self.data = data
        self.k = k
        # extract data values and labels
        self.X = data.iloc[:, :-1]
        self.y = data.iloc[:, -1]
        #self.nr_clusters = np.unique(self.y).shape[0]
        self.class_labels = np.unique(self.y)
        self.class_center = self.get_class_center()
        # get incomplete data points
        is_nan = self.X.isnull()
        inds_incomplete = is_nan.any(axis=1)
        X_incomplete = self.X.loc[inds_incomplete]
        X_complete = self.X.loc[~inds_incomplete]
        # impute data
        imputed_data = self.impute_data(X_incomplete, X_complete)
        self.X.loc[inds_incomplete] = imputed_data
        return pd.concat([self.X, self.y], axis=1)
        
    @staticmethod
    def pairwise_dist(x, y):
        # euclidean_dist
        return np.sum((x[:, None] - y) ** 2, axis=-1) ** 0.5

    def get_class_center(self):
        class_center = []
        for label in self.class_labels:
            # select inds of current label
            inds = np.where(self.y == label)[0]
            current_X = self.X.loc[inds]
            # keep only complete data points
            current_X.dropna(inplace=True)
            # get centers
            center = current_X.values.mean(axis=0)[None, :]
            class_center.append(center)
        return np.vstack(class_center)

    def impute_data(self, incomplete, complete):
        """
        incomplete: pandas DataFrame, samples with missing values
        complete: pandas DataFrame, sample with all values
        return: pandas DataFrame, with cleaned data (incomplete + complete)
        """
        complete = complete.values
        completed = np.zeros_like(incomplete)
        for i in range(incomplete.shape[0]):
            point = incomplete.values[i]
            # get missing f
            missing_f = np.where(np.isnan(point))[0]
            
            # compute distances based on present features
            distances = self.pairwise_dist(np.delete(point, missing_f, 0)[None, :],
                                           np.delete(complete, missing_f, 1))
            # get closest data points
            closest_k = np.argsort(distances)[0, :self.k]
            # replace missing feature with mean value from closest k neighbors
            point[missing_f] = complete[closest_k][:, missing_f].mean(axis=0)
            completed[i] = point
        return completed

def plot_intra_class_distances(distances, distances_sklearn, k):
    """
    Plot intra-calls distances
    distances: list, with intra class distances
    k: int, max k
    return: plot
    """
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, k+1, 1), distances, marker='o', label='Own')
    ax.plot(np.arange(1, k+1, 1), distances_sklearn, marker='x', label='Sklearn')
    ax.set_xlabel('k')
    ax.set_xticks(np.arange(1, k+1, 1), minor=True)
    ax.tick_params(axis='x', which='minor', labelbottom=True, bottom=True)
    #ax.set_xticklabels(np.arange(1, k+1, 1), minor=True)
    ax.set_ylabel('Overall intra-class distance')
    ax.legend()
    plt.show()
    fig.savefig('plots/intra_class_distances.png', bbox_inches='tight')

def compute_intra_class_distances(data, k):
    """
    Computes intra class distances when imputing data with different values for k
    data: pd DataFrame
    k: int, max k
    return: list with intra-class distance for each k
    """
    intra_class_distances = []
    intra_class_distance_sklearn = []
    for i in range(1, k + 1):
        cleaned_data = CleanData()(data, i)
        no_risk = cleaned_data.loc[cleaned_data.iloc[:, -1] == 0].iloc[:, :-1].values
        risk = cleaned_data.loc[cleaned_data.iloc[:, -1] == 1].iloc[:, :-1].values
        intra_class_distance_no_risk = CleanData.pairwise_dist(no_risk, no_risk).sum() / 2
        intra_class_distance_risk = CleanData.pairwise_dist(risk, risk).sum() / 2
        intra_class_distance_overall = intra_class_distance_no_risk + intra_class_distance_risk
        print(f'Own: {k}: {intra_class_distance_overall}')
        intra_class_distances.append(intra_class_distance_overall)
        cleaned_data = sklearn_knn(data.iloc[:, :-1], data.iloc[:, -1], i)
        no_risk = cleaned_data[data.iloc[:, -1] == 0]
        risk = cleaned_data[data.iloc[:, -1] == 1]
        intra_class_distance_no_risk = CleanData.pairwise_dist(no_risk, no_risk).sum() / 2
        intra_class_distance_risk = CleanData.pairwise_dist(risk, risk).sum() / 2
        intra_class_distance_overall = intra_class_distance_no_risk + intra_class_distance_risk
        print(f'Sklearn: {k}: {intra_class_distance_overall}')
        intra_class_distance_sklearn.append(intra_class_distance_overall)
    return intra_class_distances, intra_class_distance_sklearn

def sklearn_knn(X, y, k):
    knn = impute.KNNImputer(n_neighbors=k)
    X = knn.fit_transform(X, y)
    import pdb; pdb.set_trace()
    return X  

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='csv file with data. First line contains column names and last column class labels')
    parser.add_argument('--output', help='file name for cleaned data')
    parser.add_argument('--k', type=int, help='Impute values based on k-NearestNeighbors. If --evaluate is set its used as max k')
    parser.add_argument('--evaluate', help='Find optimal k with elbow method', default=False, action='store_true')
    args = parser.parse_args()
    data = read_data(args.input)
    if args.evaluate:
        intra_class_distances, intra_class_distance_sklearn = compute_intra_class_distances(data, args.k)
        plot_intra_class_distances(intra_class_distances, intra_class_distance_sklearn,  args.k)
    else:
        cleaned_data = CleanData()(data, args.k)
        cleaned_data.to_csv(args.output, sep=',', header=True, index=False)   
if __name__ == '__main__':
    main(sys.argv[1:])

