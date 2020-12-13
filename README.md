# KAGGLE-MOA
Mechanisms of Action Prediction

## Why is it important? 
Drugs work by interacting with specific proteins in our body. This is called (Mechanisms of action) However, researching which protein are affected for each drug cost resources and time. This is especially difficult for all the potential drugs available. Making a prediction from gene expression and cell viability would be cheaper and faster.

## Solution
Train an algorithm to predict the mechanism of action from gene expression and cell viability data. Cheaper and faster.

## Running the code
In order to get results on the test dataset it is required to run the notebook as a part of the [Kaggle competition](https://www.kaggle.com/c/lish-moa/).

### Data
All of the models require data from https://www.kaggle.com/c/lish-moa/data, more specifically the following files:

- train_features.csv
- test_features.csv
- train_targets_scored.csv
- sample_submission.csv

In addition the `path` variable needs to be changed to the location of the data files.

### Software
Running the code requires using Python3 Jupyter notebooks. The results were obtained from the Kaggle test data with hidden targets so all of the development and execution was done on Kaggle.

All of the modules require the following modules:

- numpy
- pandas
- sklearn

Feedforward Neural Network also requires:

- keras
- kerastuner
- tensorflow
- matplotlib

Gradient Boost also requires:

- lightgbm

The Ensembles requires all of the above with the exception of kerastuner.

## Used Kernels

Data Analysis:

- https://www.kaggle.com/kushal1506/moa-prediction-complete-walkthrough-eda-ensemble
- https://www.kaggle.com/amiiiney/drugs-moa-classification-eda#4-Targets-(MoA)
- https://www.kaggle.com/isaienkov/mechanisms-of-action-moa-prediction-eda

Logistic Regression:

- https://www.kaggle.com/sg1993/logistic-regression-model/notebook
- https://www.kaggle.com/barteksadlej123/basic-logistic-regression

Gradient Boosting:

- https://www.kaggle.com/nroman/moa-lightgbm-206-models
- https://www.kaggle.com/pavelvpster/moa-lgb-optuna

Neural Network:

- https://www.kaggle.com/simakov/keras-multilabel-neural-network-v1-2/notebook
- https://www.kaggle.com/elcaiseri/moa-keras-multilabel-classifier-nn-starter/notebook
- https://www.kaggle.com/gogo827jz/moa-lstm-pure-transformer-fast-and-not-bad/notebook?scriptVersionId=42679125

[Also ensemble related other source](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/)

## Methodology
We have created 3 base models: Logistic Regression, Gradient Boosting and a relatively simple Feedforward Neural Network. For every single model onehot encoding is used. Normalization seemed to affect the results of each model differently. Using PCA components worsened performance for every model in our implementation.

### Logistic Regression
file: `logreg-moa.ipynb`

We used the `sklearn` implementation of logistic regression. For logistic regression standardization offered a significant performance boost. As with most of the classical machine learning algorithm implementations, sklearn's logistic regression does not support multi-label classification. In order to overcome this we had to use sklearn's MultiOutputClassifier, which basically fits the model for each target separately. This means that we essentially created 206 models, each one predicting one of the 206 targets. This is one of the major downsides to classical machine learning methods as fitting 206 models on a dataset this size takes a considerable amount of time when compared to neural networks, especially as a lot of classical ML methods don't support GPU learning, but NNs do.

### Gradient Boosting
file: `gbm-moa.ipynb`

For Gradient Boosting we used `lightgbm`'s implementation. Similarly to logistic regression, we had to train 206 models separately for each target, but this time manually instead of using MultiOutputClassifier.

### Feedforward Neural Network
file: `fnn-moa.ipynb`

The Neural Network was implemented using `keras` and tuned with the help of `kerastuner`. The Neural Network uses 3 hidden layers, a BatchNormalization layer right after the Input layer, an output layer with 206 neurons(one for each target) and a dropout layer before each hidden and output layer. Standardization of data seemed to, at best, offer no performance benefits and possibly slightly reduces the performance of the model.

### Weighted Average Ensemble
file: `3model-weighted-average-moa.ipynb`

This model uses the results from the three models mentioned above by multiplying the results by a weight and adding them all together. The weights were chose based on the performance of the base model and experimentation.
The weights are following:

- Logistic Regression: 0.27
- Gradient Boosting: 0.33
- Neural Network: 0.4

![Weighted Averages Ensemble visualization](https://github.com/aavajanar/KAGGLE-MOA/blob/main/pics/weighted-ensemble.png?raw=true)

### Stacking Ensemble
file: `nn-gbm-logreg-nn-stacking-moa.ipynb`

This model uses the previously made FNN, GBM and Logistic Regression models by using the predictions from those models to build a new model. This is achieved by dividing the train set into 3 parts and using 2 parts to predict the third. This is done for every part and we get the train and test set predictions. We do this 3 times, once for each model, and combine all the train set predictions with each other and then all the test set predictions with each other. Then finally we build a new model, in this case a feedforward neural network, and fit it on the new train prediction set. Then we use the test predictions to predict the final target values.

![Stacking Ensemble visualization](https://github.com/aavajanar/KAGGLE-MOA/blob/main/pics/stacked-ensemble.png?raw=true)

## Results
The results were ran on both public test data and private test data on Kaggle. Performance metric is meanwise column log loss. Our initial goal was to reach 0.02 for public test data and we were hoping to reach 0.019. Results for all the models pretty much met our goals and the Weighted Average Ensemble even met our extended goal. From the results we can see that neural networks (even simple ones) seem to perform better on this dataset than classical approaches. It also seems like there is a lot of performance to be gained from ensembling different types of models together, even through a simple method like weighted averaging. We expected a bit better performance out of the Stacking Ensemble when compared to the weighted average, but the similar performance between the models could be explained by the fact that a stacking ensemble is more difficult to optimize than a weighted average ensemble so there is more potential for improving the performance.

| Model  | Private | Public |
| ------------- | ------------- | ------------- |
| Logistic Regression | 0.01780 | 0.01971 |
| Gradient Boosting | 0.01750 | 0.02028 |
| Feedforward NN | 0.01710 | 0.01968 |
| Weighted Average Ensemble | 0.01675 | 0.01901 |
| Stacking Ensemble | 0.01676 | 0.01939 |
