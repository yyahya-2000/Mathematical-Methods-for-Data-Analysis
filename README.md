# Mathematical-Methods-for-Data-Analysis, HSE course

## Abstract
This course presents the foundations of rapidly developing scientific field called intellectual data analysis or machine learning. This field is about algorithms that automatically adjust to data and extract valuable structure and dependencies from it. The automatic adjustment to data by machine learning algorithms makes it especially convenient tool for analysis of big volumes of data, having complicated and diverse structure which is a common case in modern "information era". During this course most common problems of machine learning are considered, including classification, regression, dimensionality reduction, clustering, collaborative filtering and ranking. The most famous and widely used algorithms suited to solve these problems are presented. For each algorithm its data assumptions, advantages and disadvantages as well as connections with other algorithms are analyzed to provide an in-depth and critical understanding of the subject. Much attention is given to developing practical skills during the course. Students are asked to apply studied algorithms to real data, critically analyze their output and solve theoretical problems highlighting important concepts of the course. Machine learning algorithms are applied using python programming language and its scientific extensions, which are also taught during the course. The course is designed for students of the bachelor program "Software Engineering" at the Faculty of Computer Science, HSE.

## Learning Objectives
* distinguish major problems of data analysis, solved with machine learning
* recognise and be able to apply major algorithms to solve stated problems
* understand and be able to reproduce core machine learning algorithms
* understand dependencies between algorithms, their advantages and disadvantages
* be able to use data analysis libraries from python - numpy, scipy, pandas, matplotlib and scikit-learn
* understand, which kinds of algorithms are more appropriate for what kinds of data
* know, how to transform data to make it more suitable for machine learning algorithms

## Expected Learning Outcomes
* To understand course goals
* To be familiar with key objects of the course
* To understand connection of course contents with applications
* To understand foundation of gradient approaches and the role of each component (learning rate, precondition)
* To understand basic idea of proximal updates in context of machine learning problems
* To be able to write pseudo-code for gradient decent algorithm
* To be able to derive and analyse closed form solution from scratch
* To understand concept of overfitting and crossvalidation, can derive LOOCV
* To understand and can derive properties of l1, l2 regularization
* Can derive continual learning update
* To be familiar with extension of model to GLM (Poisson, Logistic), multioutput
* To define formally problem of binary classification
* To understand geometric interpretation of linear classifier and corresponding notation
* To be able to explain why error rate cannot be used in gradient training of a classifier
* To explain the properties of different quality metrics
* To understand general idea of log reg
* To formulate optimization objective for Ligostic Regression and SVM
* To understand what is a support vector
* To understand what is a kernel and kernel trick
* To understand computational complexity of kernel methods
* To name basic components of the binary tree
* To derive impurity criterions for regression and classification problems
* To write psedocode for greedy tree construction
* To know pruning and regularization strategies
* To know how tree can be applied to unsupervisied problems
* To formulate and statistical concepts: bootstrap, bias, variance
* To understand bias-variance tradeoff in machine learning tasks
* To understand connection of machine learning algorithms and statistical methods such a bootstrap
* To derive statistical results for bootstrap and ml algorithms in simple cases
* To get idea of functional gradient and its projection
* Can derive algorithm in general and particular loss-function cases, see conections
* To understand reglarization techniques
* To understand implementation tricks
* To get familiar with variety of approaches to clustering: metric-based, graph-based and hierarchical
* Can derive updates for kmeans
* Can state optimization problem for spectral clustering
* To get familiar with anomaly detection based on convex-hull methods and disroder testing
* Can derive CUMSUM algorithm
* Can state and explain one-class SVM optimization problem
* To get familiar with EM algorithm
* To be able to make derivations for GMM
* To be able to formulate PCA as sequential optimization problem
* To be able to show solution of PCA given SVD decomposition
* To be able to formulate PCA as probabilistic model

## Course Contents
* Introduction to Machine Learning</br>
problem set-up, problem types, features, examples of application in industry
* Basic gradient optimization.</br>
Running example: Linear Regression - Vanilla gradient descent - Learning rate and pre-conditioning - Stochastic version - Proximal updates
* Linear Regression Model</br>
Close-form solution analysis and connection to MLE - Regularization and affection of closed-form solution (in orthogonal feaures case) - Regularizatin and Overfitting, basic cross-validaton - Iterative solution and stochastic version - Continual update and efficient LOOCV (Sherman-Morison update, QR) - Outlier diagnostic
* Linear Classification</br>
problem formulation - margin of a classifer - error rate loss function and its upper bounds - quality metrics for classification (Pr, Recall, F1, ROC-AUC, etc.) - Breifly define Logistic Regression
* Logistic Regression and SVM</br>
Logistic Regression as GLM - Connection to MLE - SVM in Linearly separable and non-separable cases - Connection to distance between convex sets - Support vectors analysis - Kernel trick
* Decision Trees</br>
Define binary tree (nodes, predicates, leaves) - Define Impurity Criterion - Greedy algoritm for tree construction - Pruning strategies - Regularization strategies - Decision trees for unsupervised problems and anomaly detection
* Bagging, Random Forest and Bias-Variance Tradeoff</br>
Bias-Variance Tradeoff: - Introduce concept, remind regularization - Introduce bootstrap, show on simple statistics, introduce jack-knife - Introduce bagging in machine learning algorithms Random Forest - introduce the algorithm - put in statistical context: bias-variance, margin, limit prediction distribution
* Gradient boosting</br>
Introduce general boosting algorithm - Show particular cases for regression and classification ie Adaboost, "weight" of samples interpretation - Discuss implementation details: regularization (step size, noise, step-back), second order optimization, stochastic versions - Stacking
* Clustering and Anomaly Detection</br>
Clustering - kMeans - spectral clustering - hierarchical clustering Anomaly Detection - One class SVM - CUMSUM
* EM and PCA</br>
genral formulation of EM algorithm, GMM mixture - connection with kmeans - formulation of PCA as optimization problem - solution with SVD - formulaation of PCA as probabilistic model - solution with EM algorithm
* Bayesian Linear Regression</br>
* GP for regression and classification tasks</br>
* MLP and DNN for Classification</br>
* Deep Generative Models</br>
