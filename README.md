## Welcome to the CS 7641 Project

### Introduction

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





