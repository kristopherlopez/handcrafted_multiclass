#==============================================================================
# ###########################################################
#                   Logistic Regression        
#  The following file outlines the functions used to generate
# logistic regression models:
# 
#  - binarize_labels: convert multi-class label into a matrix of binary
#   labels
#  - sigmoid: logistic function
#  - hx: hypothesis function
#  - cost: cost function with l2 regularization
#  - gradient: gradient function with l2 regularization
#  - optimize: optimize weights using scipy optimize.minimize
#  - predict: generate binary model per class and select indice of 
#  class with highest probability
# 
# ###########################################################
#==============================================================================

import numpy as np

from time import time
from scipy import optimize

class model:
    
    def __init__(self, alpha = 0):
        # L2 regularization parameter
        self.alpha = alpha
        
    def binarize_labels(self, y):
        
        return np.array((y[:,None] == np.unique(y)).astype(int))
                
    def sigmoid(self, z):
        # Logistic function
        a = 1.0 / (1.0 + np.exp(-z))
        
        # Prevents division by 0
        b = a == 1
        a[b] = 0.999999
        
        return a
        
    def hx(self, X, w):

        return self.sigmoid(np.dot(X, w))
        
    def cost(self, w):
        # Cost function
        y0 = self.y * np.log(self.hx(self.X, w))
        y1 = (1 - self.y) * np.log(1 - self.hx(self.X, w))
        
        # Cost regularization component
        c = (-1.0 / self.m) * np.sum(y0 + y1) + self.alpha / (2 * self.m) * np.sum(w[1::] ** 2)
        #print('lr cost %0.3f' %c)
        
        return c
        
    def gradient(self, w):
        # Gradient function
        g = self.hx(self.X, w) - self.y
        g = (1.0 / self.m) * np.dot(g, self.X)
        
        # Gradient regularization component
        g[1::] += self.alpha / self.m * w[1::]

        return g
        
    def optimize(self, iterations = 20):
        # Find optimal weights that minimizes the cost function
        optim_w = optimize.minimize(
            fun = self.cost, 
            x0 = np.ones(self.X.shape[1]), 
            args = (), 
            method = 'L-BFGS-B', 
            jac = self.gradient,
            options = {'maxiter': iterations}
            )
                                    
        return optim_w.x
        
    def predict(self, X_train, X_test, y_train):
        print('### Logistic Regression ###')
        print('# Regularization: ' + str(self.alpha) + ' #')

        start = time()
        # Add intercept
        self.X = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        self.X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        
        # Split single column multi-class label into matrix of binary labels
        self.y_train = self.binarize_labels(y_train)
        
        self.m = X_train.shape[0]
        
        # Initiate matrix for predicted probabilities
        pred_x = np.zeros((self.y_train.shape[1], self.X_test.shape[0]), dtype=float)

        # Generate logistic regression for each class and store probabilities
        for i in range(self.y_train.shape[1]):
            print('Training model for class: ' + str(i))
            self.y = np.transpose(self.y_train)[i]
            o_w = self.optimize(iterations = 50)
            pred_x[i] += self.hx(self.X_test, o_w)
        
        train_time = time() - start        
        print ("took %0.3fs to train multi-class model" %train_time)
        
        # Return indice of class with highest probability for each test observation
        start = time()        
        y_hat = np.argmax(np.transpose(pred_x), axis=1)
        test_time = time() - start        
        print ("took %0.3fs to get predictions" %test_time)
                
        return y_hat

#==============================================================================
# ###########################################################
#                         Bibliography                               
# The functions in this file took inspiration from: 
# 
#  - (Divakar, 2016)                          
#  - (FindBoat, 2012)                            
#  - (Fg Nu, 2012)
#
# for further detail please refer to the report bibilography                                      
# ###########################################################
#==============================================================================