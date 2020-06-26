#!/usr/bin/env python
# coding: utf-8

# In[29]:


##############################################################################
# Set up
import numpy as np
import pandas as pd

# Load in data
X = pd.read_csv(r"C:\Users\Elianna\Documents\2020 Summer\ML\Project/datasets_222487_478477_framingham-cleanonly.csv")

###############################################################################
# Mean Shift 
from sklearn.cluster import MeanShift, estimate_bandwidth

bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=4000)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print("-----Mean Shift-----")
print("Cluster Centers")
print(cluster_centers)
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)
