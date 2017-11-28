#==============================================================================
# ###########################################################
#                   Naive Bayes        
#  The following file outlines the functions used to generate
# naive bayes models:
# 
#  - summarize: creates dictionary of counts, sums and probabilities
#  - predict: combines prior and likelihoods across classes and selects
#  indice of class with highest probability
# 
# ###########################################################
#==============================================================================

import numpy as np

from time import time


class model:
    
    def __init__(self, prior_weight = 0):
        
        self.prior_weight = prior_weight
        
    def summarize(self):
        
        # Initiate dictionary to store counts, sums and probabilities
        dict = {}
        for l in np.unique(self.y):
            dict[l] = {
                'class_tf_idf': 0, 
                'class_tf_idf_probabilities': 0,
                'class_feature_tf_idf': np.zeros(self.X.shape[1]),
                'class_feature_tf_idf_probabilities': np.zeros(self.X.shape[1]),
                'class_feature_count': np.zeros(self.X.shape[1]),
                'class_feature_sum': np.zeros(self.X.shape[1]),
                'class_feature_mean': np.zeros(self.X.shape[1]),
                'class_feature_stdev': np.zeros(self.X.shape[1])
            }       
        
        # Store class and intra-class feature sums
        for i in range(self.X.shape[0]):
            dict[self.y[i]]['class_tf_idf'] = dict[self.y[i]]['class_tf_idf'] + np.sum(self.X[i])
            dict[self.y[i]]['class_feature_tf_idf'] = dict[self.y[i]]['class_feature_tf_idf'] + self.X[i]
       
        # Store class and intra-class log probabilities
        for l in np.unique(self.y):
            dict[l]['class_tf_idf_probabilities'] = np.log(dict[l]['class_tf_idf']/np.sum(self.X))
            dict[l]['class_feature_tf_idf_probabilities'] = np.log((dict[l]['class_feature_tf_idf'] + 1)/(dict[l]['class_tf_idf'] + self.X.shape[1]))
        
        dict['feature_matrix'] = np.zeros(shape=(30, self.X.shape[1]))
        dict['class_prior_vector'] = np.zeros(shape=(30, 1))
    
        # Prepare matrix of class conditional probabilities and prior probabilities to be used for prediction
        for l in np.unique(self.y):
            dict['feature_matrix'][int(l)] = dict[l]['class_feature_tf_idf_probabilities']
            dict['class_prior_vector'][int(l)] = dict[l]['class_tf_idf_probabilities']
             
        return dict
        
    def predict(self, X_train, X_test, y_train):
        print('### Naive Bayes ###')
        print('# Prior weight: ' + str(self.prior_weight) + ' #')
        
        self.X = X_train
        self.y = y_train
        
        # Create dictionary for prediction
        start = time()
        dict = self.summarize()
        end = time() - start
        print ("took %0.3fs to prepare dict " %end)
        
        X_fm = dict['feature_matrix']
        
        # Apply weight to prior probabilities
        X_fm = X_fm + self.prior_weight * dict['class_prior_vector']
        
        # Return indice of class with highest probability for each test observation
        start = time()
        y_hat = np.argmax(np.matmul(X_test, np.transpose(X_fm)), axis=1)
        test_time = time() - start
        print ("took %0.3fs to predict" %test_time)
        
        return y_hat

#==============================================================================
# ###########################################################
#                         Bibliography                               
# The functions in this file took inspiration from:
#
#  - (Brownlee, 2016)
#
# for further detail please refer to the report bibilography                                      
# ###########################################################
#==============================================================================