#!/usr/bin/env python
import pandas as pd
import numpy as np
import argparse
import sys

def read_data(path_to_data):
    #read data
    return pd.read_csv(path_to_data, sep=',', header=0)

class CleanData:
    def __init__(self, data, k):
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
        imputed_data = self.impute_data(X_incomplete, X_complete)
        self.X.loc[inds_incomplete] = imputed_data
        self.cleaned_data = pd.concat([self.X, self.y], axis=1)
        
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
        
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='csv file with data. First line contains column names and last column class labels')
    parser.add_argument('--output', help='file name for cleaned data')
    parser.add_argument('--k', type=int, help='Impute values based on k-NearestNeighbors')
    args = parser.parse_args()
    data = read_data(args.input)
    cleaned_data = CleanData(data, args.k)
    cleaned_data.cleaned_data.to_csv(args.output, sep=',', header=True, index=False)   
if __name__ == '__main__':
    main(sys.argv[1:])
