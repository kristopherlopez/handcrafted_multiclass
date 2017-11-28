#==============================================================================
# ###########################################################
#                        Output                             
#  This file contains the functions used for performing
# singular value decomposition:
# 
#  + dimensionality_reduction:
#       - takes retained variance, training data and corresponding test
#       data as inputs
#       - decomposes using scipy linalg and returns reduced train and
#       test data
#  + sparse dimensionality reduction:
#       - similar to above excep uses scipy sparse linalg
# 
# ###########################################################
#==============================================================================

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from time import time

def deconstruct(X_train):
    print('### Singular Value Decomposition ###')   
   
    # Perform a full singular value decomposition and assign the results to U, s and Vt
    start = time()
    U, s, Vt = sp.linalg.svd(X_train, full_matrices = False)
    end = time() - start
    print("took %0.3fs to perform SVD" %end)
    
    return U, s, Vt

def reconstruct(U, s, Vt, X_test, var):
    print('# Retained variance: ' + str(var) + ' #')
    # Set the required proportion of variance to retain
    retained_var_p = var
        
    S = np.diag(s)
    V = np.transpose(Vt)

    # Information graph showing how much variance is contributed by each successive eigenvector
#==============================================================================
#     plt.figure(1, figsize = (4, 3))
#     plt.clf()
#     plt.axes([.2, .2, .7, .7])
#     plt.plot(s / sum(s), linewidth = 2)
#     plt.axis('tight')
#     plt.xlabel('n_components')
#     plt.ylabel('explained_variance_ratio')
#==============================================================================

    # Return only the first k components, such that the retained variance proportion is above the specified level
    start = time()
    var = 0
    for i in range(len(s)):
        var += s[i]
        if var / sum(s) >= retained_var_p:
            break
        
    # Return the k components, transformed into k features for the training set
    # Apply the same transformation to the test set
    X_train_SVD = U[:,:i].dot(S[:i,:i])
    X_test_SVD = X_test.dot(V[:,:i])

    end = time() - start
    print ("%0i components utilised" %i)
    print ("took %0.3fs to apply transformation" %end)
    
    return X_train_SVD, X_test_SVD

def sparse_dimensionality_reduction(X_train, X_test, pc):
    print('### Sparse Singular Value Decomposition ###')
    print('# Principal Components: ' + str(pc) + ' #')
    
    # Perform singular value decomposition and assign the results to U, s and Vt
    start = time()
    U, s, Vt = sp.sparse.linalg.svds(X_train, k = pc, which = 'LM')
    end = time() - start
    print("took %0.3fs to perform Sparse SVD" %end)
    
    S = np.diag(s)
    V = np.transpose(Vt)

    # Return the k components, transformed into k features for the training set
    # Apply the same transformation to the test set
    X_train_SVD = U.dot(S)
    X_test_SVD = X_test.dot(V)

    end = time() - start
    print ("took %0.3fs to apply transformation " %end)
    return X_train_SVD, X_test_SVD

