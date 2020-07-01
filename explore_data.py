#!/usr/bin/env python
################################################
#Georgia Institute of Technology
#Aaron Pfennig
#2020
################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from clean_data import CleanData

def read_data(path_to_data):
    #read data
    return pd.read_csv(path_to_data, sep=',', header=0)

class Visualizations:
    def __init__(self, data, inds_incomplete):
        """
        Initialize visualization class.
        data: pandas DataFrame, contains data, samples in rows features in columns, last column contains class labels
        inds_complete: boolean, pandas Series, indicating whether sample had missing features or not
        """
        self.data = data
        self.inds_incomplete = inds_incomplete
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

    def plot_histograms(self):
        """
        Plot histograms of features
        return: d plots
        """
        X_complete = self.X.loc[~self.inds_incomplete, :]
        y_complete = self.y.loc[~self.inds_incomplete]
        X_incomplete = self.X.loc[self.inds_incomplete, :]
        y_incomplete = self.y.loc[self.inds_incomplete]
        inds_no_risk_incomplete = np.where(y_incomplete == 0)[0]
        inds_risk_incomplete = np.where(y_incomplete == 1)[0]
        inds_no_risk_complete = np.where(y_complete == 0)[0]
        inds_risk_complete = np.where(y_complete == 1)[0] 
        # iterate over all features
        for f in self.X.columns:
            min_val = np.floor(self.X.loc[:, f].min())
            min_val -= min_val/10
            max_val = np.ceil(self.X.loc[:, f].max())
            max_val += max_val/10 
            fig, ax = plt.subplots()
            if self.X.loc[:, f].unique().shape[0] <= 3:
                nr_bins = [-0.1, 0.5,  1.1]
                xtick_labels = [0, 1]
            else:
                nr_bins = self.X.loc[:, f].unique().shape[0] // 2
                xtick_labels = np.arange(min_val, max_val, (max_val - min_val) / nr_bins)
            ax.hist(X_complete.loc[inds_no_risk_complete, f], bins=nr_bins, align='mid',
                    histtype='stepfilled', color='green', alpha=0.5, label='No risk complete')
            ax.hist(X_complete.loc[inds_risk_complete, f], bins=nr_bins, align='mid',
                    histtype='stepfilled', color='red', alpha=0.5, label='Risk complete')
            # imputed data
            ax.hist(X_incomplete.loc[inds_no_risk_incomplete, f], bins=nr_bins, align='mid', linewidth=2,
                    histtype='step', color='green', label='No risk incomplete')
            ax.hist(X_incomplete.loc[inds_risk_incomplete, f], bins=nr_bins, align='mid', linewidth=2,
                    histtype='step', color='red', label='Risk incomplete')            
            ax.set_xlabel(f)

            ax.set_ylabel('Occurrences')
            ax.legend(bbox_to_anchor=(0.5, -0.13), loc='upper center',  ncol=2)
            fig.savefig(f'plots/histograms/{f}.png', bbox_inches='tight')
            plt.close()
            
    def plot_feature_correlations(self):
        """
        Plot two features against each other
        """
        features = self.X.columns
        feature_pairs = []
        # get all possible pairs of features
        inds_no_risk = np.where(self.y == 0)[0]
        inds_risk = np.where(self.y == 1)[0]

        for feature1 in features:
            for feature2 in features:
                if feature1 == feature2:
                    continue
                sorted_features = sorted((feature1, feature2))
                pair = (sorted_features[0], sorted_features[1])
                if not pair in feature_pairs:
                    feature_pairs.append(pair)
        for pair in feature_pairs:
            fig, ax = plt.subplots()
            ax.scatter(self.X.loc[inds_no_risk, pair[0]], self.X.loc[inds_no_risk, pair[1]], marker='x', color='green', label='No risk', alpha=0.5)
            ax.scatter(self.X.loc[inds_risk, pair[0]], self.X.loc[inds_risk, pair[1]], marker='o', color='red', label='Risk', alpha=0.5)
            ax.set_xlabel(pair[0])
            ax.set_ylabel(pair[1])
            ax.legend(bbox_to_anchor=(0.5, -0.13), loc='upper center',  ncol=2)
            fig.savefig(f'plots/scatterplots/{pair[0]}_{pair[1]}.png', bbox_inches='tight')
            plt.close()
            
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='csv file with data. First line contains column names and last column class labels')
    args = parser.parse_args()
    data = read_data(args.input)
    inds_incomplete = data.isnull().any(axis=1)
    data = CleanData()(data, k=3)
    visualizations = Visualizations(data, inds_incomplete)
    visualizations.plot_histograms()
    visualizations.plot_feature_correlations()
    


if __name__ == '__main__':
    main(sys.argv[1:])
