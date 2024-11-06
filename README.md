# IN1102 Machine Learning

This repository was created for the machine learning course, which is part of the Computer Science postgraduate program at CIn/UFPE. It contains projects focused on training, evaluating and comparing various classifiers.

## Description

The dataset used to develop this project was the from the UCI Machine Learning dataset called "SPECTF Heart".  

The assignment is divided between two questions, detailed below.

### First Question

1. Consider the "SPECTF Heart" data from the UCI website (http://archive.ics.uci.edu/dataset/96/spectf+heart). Concatenate the SPECTF.test and SPECTF.train datasets to form the SPECTF data set with 267 individuals (lines) described by 45 variables (columns), the first column being the class variable. *Parameters*: T = 100; $$ \in = 10^-6$$; m = 1.1.
   
    1.1 Run the KFCM-K and KFCM-K-W.1 algorithms 50 times each to get a fuzzy partition with $$c\in{2,3,4,5}$$. For each c select the best result according to the objective function. For each c get the corresponding crisp partition from the best fuzzy partition. For each c and crisp partition calculate the silhouette (Sil). Plot $$Sil x c$$ for c$$ \in $${2,3,4,5} and choose the number of clusters: $$c^* = \arg \max_c \text{Sil}(c)$$.

    1.2 For each algorithm and best fuzzy partition with $$ c^*$$ , calculate the Modified partition coefficient. Comment.
    
    1.3 For each algorithm and crisp partition corresponding to the best fuzzy partition with $$ c^*$$, calculate the corrected Rand index. Comment.

    1.4 For each algorithm and best result according to the objective function with $$ c^*$$, show: 
    i. the prototypes of each group (g1,...,gc); 
    ii. the vector of width parameters of each group (s1,..., sc) 
    iii. the confusion matrix of the crisp partition versus the a priori partition; 
    iv. the plot of the objective function versus the iterations;
Reference for the KFCM-K and KFCM-K-W.1 algorithms: **Gaussian Kernel Fuzzy C-Means with Width Parameter Computation and Regularization**,
https://doi.org/10.1016/j.patcog.2023.109749

### Second Question
2. Consider once again the "SPECTF" dataset with two a priori classes.
   1. Use "30 10-folds" stratified cross validation to evaluate and compare the 5 classifiers: i) Bayesian Gaussian, ii) k-neighbor based Bayesian, iii) Bayesian based on the Parzen window, iv) logistic regression, v) using a rule of majority vote from the first 4 classifiers. 
   2. When necessary, cross-validate 5-folds on the remaining 9 folds to do hyperparameter fine-tuning. Then, retrain the model with the remaining 9-folds using the optimal hyperparameters. Use stratified sampling.
   3. Get a point estimate and confidence interval for each classifier evaluation metric (Error rate, accuracy, coverage, F-measure);
   4. Use the Friedman test (non-parametric test) to compare the classifiers, and the post test (Nemenyi test), for each of the metrics. 
   5. For each assessment metric, plot the learning curve for the Gaussian Bayesian classifier. Using stratified sampling, use 70% of the data for training and 30% for testing. Train the algorithm with sets training from 5% to 100% of the original training set, with step of 5% (using stratified sampling). Comment.
   
**Observations**
- For the k-NN algorithm, consider the Euclidean, City-Block and Chebishev distances to define the neighbourhood. Use cross-validation techniques to fixate the parameters k and the distance.

