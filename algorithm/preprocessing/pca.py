#==============================================================================
# ###########################################################
#                        Output                             
#  This file contains the functions used for performing
# principal components analysis:
# 
#  + dimensionality_reduction:
#       - takes retained variance, training data and corresponding test
#       data as inputs
#       - centres data, decomposes using scipy linalg and returns 
#       reduced train and test data
# 
# ###########################################################
#==============================================================================

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

def dimensionality_reduction(X_train, X_test, var):
    print("Principal Components Analysis")

    # Set the required proportion of variance to retain
    retained_var_p = var

    # Centre the training data set (and record means for centering test data)
    X_means = np.mean(X_train, axis = 0)
    X_train_centre = np.zeros(np.shape(X_train))
    X_test_centre = np.zeros(np.shape(X_test))
    for i in range(np.shape(X_train)[1]):
        X_train_centre[:,i] = X_train[:,i] - X_means[i]
        X_test_centre[:,i] = X_test[:,i] - X_means[i]

    # Perform a full singular value decomposition and assign the results to U, s and Vt
    start = time.time()
    U, s, Vt = sp.linalg.svd(X_train_centre, full_matrices = False)
    end = time.time()
    runTime = end - start
    print("%0.3f s to perform SVD" %runTime)
    
    S = np.diag(s)
    V = np.transpose(Vt)

    # Information graph showing how much variance is contributed by each successive eigenvector
    plt.figure(1, figsize = (4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(s / sum(s), linewidth = 2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio')

    # Return only the first k components, such that the retained variance proportion is above the specified level
    start = time()    
    var = 0
    for i in range(len(s)):
        var += s[i]
        if var / sum(s) >= retained_var_p:
            break

    # Return the k components, transformed into k features for the training set
    # Apply the same transformation to the test set
    X_train_PCA = U[:,:i].dot(S[:i,:i])
    X_test_PCA = X_test_centre.dot(V[:,:i])
    
    end = time() - start
    print ("took %0.3fs to find components and apply transformation " %end)
    
    return X_train_PCA, X_test_PCA