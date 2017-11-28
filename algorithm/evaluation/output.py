#==============================================================================
# ###########################################################
#                        Output                             
#  This file contains the functions used for generating pred-
# ictions on the training and test data:
# 
#  + predict_on_training_quick:
#       - trains a logistic regression model with 0.55 l2 regularization
#       - trains on 90% of the training data and tests on remaining 10%
#  + predict_on_training_long:
#       - trains a logistic regression model with 0.55 l2 regularization on
#       SVD retaining 90% variance
#       - trains a naive bayes model with 0 prior weight
#       - trains a nearest neighbour model with 20 neighbours on SVD
#       retaining 40% variance
#       - trains an voting rule ensemble model with:
#           4 votes from the logistic regression model
#           2 votes from the naive bayes model
#           3 votes from the nearest neighbour model
#       - trains on 90% of the training data and tests on remaining 10%
#  + predict_on_test_quick:
#       - trains same model as predict_on_training_quick
#       - trains on 100% of the training data and outputs to predicted_labels
#  + predict_on_test_long:
#       - trains same model as predict_on_training_long
#       - trains on 100% of the training data and outputs to predicted_labels
# 
# ###########################################################
#==============================================================================

import csv

from preprocessing import data
from preprocessing import svd
from preprocessing import pca
from classifiers import logistic_regression as lr
from classifiers import naive_bayes as nb
from classifiers import nearest_neighbours as nn
from classifiers import ensemble_model as em
from evaluation import metrics

def predict_on_training_quick(paths):
    
    # Get paths
    data_path = paths['training_data']
    labels_path = paths['training_labels']    
    
    # Get and split data    
    training_data = data.get_training_data(data_path, labels_path)
    X_train, X_test, y_train, y_test, z_list = data.split_data(training_data, 0.1)

    
    # Initiate model and predict
    y_hat_lr = lr.model(alpha = 0.55).predict(X_train, X_test, y_train)

    # Evaluate model
    cf_lr = metrics.confusionMatrix(y_hat_lr, y_test, 'logistic_regression', paths)
    
    print('## Logistic Regression results ##')
    
    print(metrics.accuracy(cf_lr))
    print(metrics.precision(cf_lr))
    print(metrics.recall(cf_lr))
    print(metrics.fscore(cf_lr))
    
def predict_on_training_long(paths):
    
    # Get paths
    data_path = paths['training_data']
    labels_path = paths['training_labels']      
    
    # Get and split data
    training_data = data.get_training_data(data_path, labels_path)
    X_train, X_test, y_train, y_test, z_list = data.split_data(training_data, 0.05)
    
    # Release variable
    training_data = None

    # Decompose training data into singular values
    U, s, Vt = svd.deconstruct(X_train)
    
    # Build logistic regression with 90% variance retained
    X_train_reduced, X_test_reduced = svd.reconstruct(U, s, Vt, X_test, 0.9)
    y_hat_lr = lr.model(alpha = 0.55).predict(X_train_reduced, X_test_reduced, y_train)
    
    # Build nearest neighbours with 40% variance retained
    X_train_reduced, X_test_reduced = svd.reconstruct(U, s, Vt, X_test, 0.4)
    y_hat_nn = nn.model(k = 20).predict(X_train_reduced, X_test_reduced, y_train)
    
    # Build naive bayes
    y_hat_nb = nb.model(prior_weight = 0).predict(X_train, X_test, y_train)

    # Create ensemble model
    y_hat_em = em.model(y_hat_lr, y_hat_nb, y_hat_nn, 4, 2, 3)

    # Evaluate models
    cf_lr = metrics.confusionMatrix(y_hat_lr, y_test, 'logistic_regression', paths)
    cf_nb = metrics.confusionMatrix(y_hat_nb, y_test, 'naive_bayes', paths)
    cf_nn = metrics.confusionMatrix(y_hat_nn, y_test, 'nearest_neighbours', paths)
    cf_em = metrics.confusionMatrix(y_hat_em, y_test, 'ensemble_model', paths)
    
    print('## Logistic Regression results ##')
    
    print(metrics.accuracy(cf_lr))
    print(metrics.precision(cf_lr))
    print(metrics.recall(cf_lr))
    print(metrics.fscore(cf_lr))
    
    print('## Naive Bayes results ##')
    
    print(metrics.accuracy(cf_nb))
    print(metrics.precision(cf_nb))
    print(metrics.recall(cf_nb))
    print(metrics.fscore(cf_nb))
    
    print ('## Nearest Neighbours results ##')
    
    print(metrics.accuracy(cf_nn))
    print(metrics.precision(cf_nn))
    print(metrics.recall(cf_nn))
    print(metrics.fscore(cf_nn))
    
    print ('## Ensemble Model results ##')
    
    print(metrics.accuracy(cf_em))
    print(metrics.precision(cf_em))
    print(metrics.recall(cf_em))
    print(metrics.fscore(cf_em))
    
    return
    
def predict_on_test_quick(paths):
    
    # Get paths
    training_path = paths['training_data']
    labels_path = paths['training_labels']
    test_path = paths['test_data']
    results_path = paths['predicted_labels']

    # Get and split data
    training_data = data.get_training_data(training_path, labels_path)
    X_train, y_train = data.split_data(training_data)
    X_test, X_labels = data.get_test_data(test_path) 

     
    # Initiate model and predict     
    y_hat_lr = lr.model(alpha = 0.55).predict(X_train, X_test, y_train)

    # Convert numerical labels back to alphanumeric
    y_hat_lr = data.convert_labels(y_hat_lr)

    # Populate predicted_labels.csv
    print('### Write to predicted labels ###')
    ofile = open(results_path, 'w', newline = '')
    ofile.truncate()
    output = csv.writer(ofile)
    
    for i in range(X_labels.shape[0]):
        output.writerow([
            X_labels[i][0],
            y_hat_lr[i][0]
        ])
    
    ofile.close()
    
    return
    
def predict_on_test_long(paths):
    
    # Get paths
    training_path = paths['training_data']
    labels_path = paths['training_labels']
    test_path = paths['test_data']
    results_path = paths['predicted_labels']
    
    # Get and split data
    training_data = data.get_training_data(training_path, labels_path)
    X_train, y_train = data.split_data(training_data)
    X_test, X_labels = data.get_test_data(test_path)
    
     # Release variable
    training_data = None

    # Decompose training data into singular values
    U, s, Vt = svd.deconstruct(X_train)

    # Build logistic regression with 90% variance retained
    X_train_reduced, X_test_reduced = svd.reconstruct(U, s, Vt, X_test, 0.9)
    y_hat_lr = lr.model(alpha = 0.55).predict(X_train_reduced, X_test_reduced, y_train)
    
    # Build nearest neighbours with 40% variance retained
    X_train_reduced, X_test_reduced = svd.reconstruct(U, s, Vt, X_test, 0.4)
    y_hat_nn = nn.model(k = 20).predict(X_train_reduced, X_test_reduced, y_train)
    
    # Build naive bayes
    y_hat_nb = nb.model(prior_weight = 0).predict(X_train, X_test, y_train)

    # Create ensemble model
    y_hat_em = em.model(y_hat_lr, y_hat_nb, y_hat_nn, 4, 2, 3)
    
    # Convert numerical labels back to alphanumeric
    y_hat_em = data.convert_labels(y_hat_em)

    # Populate predicted_labels.csv
    ofile = open(results_path, 'w', newline = '')
    ofile.truncate()
    output = csv.writer(ofile)
    
    for i in range(X_labels.shape[0]):
        output.writerow([
            X_labels[i][0],
            y_hat_em[i][0]    
        ])
    
    ofile.close()
    
    return

#==============================================================================
# ###########################################################
#                         Bibliography                               
# The functions in this file took inspiration from: 
# 
#  - (scikit-learn, 2017)                        
# 
# for further detail please refer to the report bibilography                                      
# ###########################################################
#==============================================================================