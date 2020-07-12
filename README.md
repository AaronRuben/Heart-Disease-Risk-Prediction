## Welcome to the CS 7641 Project

### Introduction

### Background 
Cardiovascular diseases (CVD) are the number 1 cause of deaths worldwide with over 17 million deaths per year [1]. Since 1947, the Framingham Heart Study (FHS) has been a central pioneer of the expansion from treatment-based to preventive medicine by establishing risk factor determination as a central pillar of data analysis of studying disease. [2]. With more data available than ever before, machine learning has been shown to improve risk score predictions for CVD and beyond [3]. 
The dataset consists of continuous (ie Age, Cigarettes per day, etc) and binary (Is this person a current smoker) variables. In total there are 15 features included by over 4000 records. The aim is to predict if someone is in risk of developing a CVD within the next ten years and to pinpoint potential risk factors. The dataset is publicly available on [kaggle](https://www.kaggle.com/dileep070/heart-disease-prediction-using-logistic-regression).

### Data Cleaning
One of the most challenging tasks in data analysis is data cleaning on which scientists spend an enourmous amount of time in enhancing the reliability and quality of the data. In working with any real dataset, usually there are some datapoints in which some features are missing. This is usually due to not maintaining the data propoerly or becuase this data has not been recorded initially. These missing values incur errors and unreliability in the analysis and eventually leads in not having a robust predictive model. To overcome this problem, we need to find and imput values for the missing ones by using the common probabilistic models or even refine the data to remove the errors. This helps us to obtain a reliable dataset which improves the quality of the training data for analysis and provides an accurate decision-making.
The data used in the current project has 4240 labeled datapoints with 15 features of which 6 features have some missing values. Similar to all experimental set of data, the data used in this project has some missing values for some features. There are many logics according to which data cleaning can be executed. In the simplest way, one could simply remove the datapoints with any number of missing values. In this case, by doing so we could lose significant portion of the data which could diminish the reliability of the model. In a more efficient way, one could find the average of values of a feature and imput that for all missing values for the corresponding feature. Or if the labels are given, like in this project, missing features for each datapoint can be calculated by finding the average of the features for that associated label. In this project, the dataset contains two labels which corresponds to whether the person develops heart disease or not. So, we bascialy had two different ways to go. First way was to imput the missing values by using clustering method of the unsupervised data. In this method, the dataset is divided into different clusters according to the features that do not have any missing value. Then, the missing values of each datapoint is found by averaging the values of the corresponsing feature of that cluster. In this method, the best number of clusters is determined by using the Elbow method. The other method was to find the missing values based on the labels. To do so, the *k*-nearest-neighbors within the class were identified and their average value for each missing feature is calculated. The optimal number for *k* has been determined using the elbow method. The plot below shows that *k=6* is a suitable choice as well as that the KNNImputer implented in scikit-learn and our own implementation yield similar results. In this project, although the differences between the calculated values are not very huge, imputing based on the labels was used.

![Elbow method kNN](https://github.com/AaronRuben/Heart-Disease-Risk-Prediction/blob/AP/plots/intra_class_distances.png)


### Finding the best estimator
We evaluate the performance of logistic regression (LR), support vector machine (SVC), random forest (RF) and a simple neural network (NN) with respect to prediction of risk of developing a heart disease within ten years. 
The performances have been assed in terms of accuracy (ACC) and area under (AUC) the receiver operating characteristic (ROC) curve. These to metrics did not positively correlate with each other since the present dataset is highly skewed. The number of patients without risk of developing a heart disease is dramatically outweighing those who are in risk. Hence, the classifier would perform very well when always predicting *No risk* when using accuracy as a measure. However, this would cause a drop in true positive rate or sensitivity and thereby lead to a low AUC. Thus, the same classifier would perform badly in terms AUC.

In order to decide which of the above mentioned classifier works the best on our dataset, we performed a grid search over the following parameter space:
 - k the number of nearest neighbors within a class to consider when imputing the missing values
- d: the degree used for creating polynomial features
- n : the number of components to keep during the PCA
- method: the estimator (one of LR, SVC, RF, NN)

When assessing the performs with respect to the accuracy a random forest classifier outperforms the other classifier with an accuracy of *0.86* but the AUC is only *0.62*. *k* has been found to be 2, *d* to be 3 and *n* to be 4. However, as shown below the scenario of high accuracy and low AUC described above occurs with these settings. The model just always predicts *No risk* which is pretty good guess just by chance but it misses nearly all patients who are at risk. In a medical setting this is particularly bad thus, AUC is the more appropriate metric to assess the classifiers performance.

![Confusion matrix random forest](https://github.com/AaronRuben/Heart-Disease-Risk-Prediction/blob/master/output/confusion_matrix_rf.png "Confusion matrix RF") ![ROC curve RF](https://github.com/AaronRuben/Heart-Disease-Risk-Prediction/blob/master/output/roc_curve_rf.png "ROC curve RF")

When doing the grid-search with respect to the AUC, the SVC turns out to outperform all other classifiers. With *k=5*, *d=3* and *n=300* it achieves an AUC of *0.71* and accuracy of *0.80*. But as can be seen below the number of correctly predicted patients is higher albeit not greater either. 

![Confusion matrix SVC](https://github.com/AaronRuben/Heart-Disease-Risk-Prediction/blob/master/output/confusion_matrix_svc.png) ![ROC curve SVC](https://github.com/AaronRuben/Heart-Disease-Risk-Prediction/blob/master/output/roc_curve_svc.png)

When the analysis is done without a PCA the AUC remains at *0.71*, the accuracy drops to *0.75* but the number true positive (TP) predictions increases as well as the number in false positive (FP) predictions.

![Confusion matrix SVC without PCA](https://github.com/AaronRuben/Heart-Disease-Risk-Prediction/blob/master/output/confusion_matrix_svc_without_pca.png)


### Dependencies

 - scikit-learn v0.23.1
 - pandas
 - numpy
 - joblib
 - matplotlib 
 - keras
 - tqdm

### How to run the pipeline
    ./pipeline.py
    Required parameters 
	    --data <path_to_data_file> csv file, column names in first line, last column contains class labels
	    --output_dir <path_to_output_directory>
    
    Optional parameters
	    -k <int> number of k-NearestNeighbor to consider while imputing values, default=5
	    --degree <int> max degree for creating polynomial features, default=3
	    --n_components <int> number of components to keep during PCA, default=300
	    --method [RF, LR, SVC, NN] supervised learning method to employ, default='SVC'
	    --categorical column names of categorical features, separated by a space, default=['education', 'male', 'currentSmoker', 'prevalentStroke', 'prevalentHyp', 'diabetes']
	    --threads <int> number of threads to use where parallelization is possible, default=8
	    --optimize run Grid search to find best parameters. Modify parameter space at top of script in perform_gridsearch()
	    --verbose verbosity
    
    Expected Output in --output_dir:
    - model.sav or model.h5 trained model
    - roc_curve.png Plot of ROC curve of evaluation during cross validation on training set
    - confusion_matrix.png Confusion matrix of prediction on test set
    - correlation_matrix.png Correlation matrix of raw feature after missing value imputation
    - pca_transformed.png Plot of datapoints in first 3 components
    - pca_recovered_variance.png Cumulative plot variance recovered with kept components
    - if --method RF
	    - RF_best_features.tab 10 best features of RF

