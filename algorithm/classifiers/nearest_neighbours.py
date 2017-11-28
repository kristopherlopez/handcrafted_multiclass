#==============================================================================
# ###########################################################
#                   Nearest Neighbours        
#  The following file outlines the functions used to generate
# nearest neighbour models:
# 
#  - distance: calculates euclidean distance between two points
#  - predict: stores distance between a test observation and all
#   training observations, returns k nearest points and selects class
#   with highest count
# 
# ###########################################################
#==============================================================================


import numpy as np

from time import time

class model:
    
    def __init__(self, k = 50):
        
        self.k = k
        
    def distance(self, a, b, q = 2):
        # Minkowski distance (Euclidean distance when q = 2)
        return np.sum((a-b)**q, axis=1)**(1.0/q)
        
    def predict(self, X_train, X_test, y_train):
        print('### Nearest Neighbours ###')
        print('# k: ' + str(self.k) + ' #')
        
        # Find k nearest neighbours
        start = time()
        nn = np.array([np.argpartition(self.distance(X_train, X), self.k)[:self.k] for X in X_test])
        train_time = time() - start
        print ("took %0.3fs to discover k nearest neighbours" %train_time)
        
        start = time()
        y_hat = []
        
        # Find neighbour with most frequent predicted label
        for i in range(nn.shape[0]):
            labels = []
            for n in nn[i]:
                labels.append(y_train[n])
            y_hat.append(np.bincount(labels).argmax())
        
        test_time = time() - start
        print ("took %0.3fs to predict" %test_time)
        
        return y_hat

#==============================================================================
# ###########################################################
#                         Bibliography                               
# The functions in this file took inspiration from: 
# 
#  - (blue, 2016)                          
#  - (JoshAdel, 2011)
#
# for further detail please refer to the report bibilography                                      
# ###########################################################
#==============================================================================