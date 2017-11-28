#==============================================================================
# ###########################################################
#                   Logistic Regression        
#  The following file outlines the functions used to create
# ensemble models:
# 
#  - model: takes three models and applies integer weights for
#   class prediction
# 
# ###########################################################
#==============================================================================

import numpy as np


def model(m1, m2, m3, n1, n2, n3):
    print('### Three model Ensemble ###')
    print('# Model 1: ' + str(n1) + ' votes')
    print('# Model 2: ' + str(n2) + ' votes')
    print('# Model 3: ' + str(n3) + ' votes')

    a = np.tile(m1, (n1 + 1, 1))
    b = np.tile(m2, (n2 + 1, 1))
    c = np.tile(m3, (n3 + 1, 1))
    
    y_hat_ec = np.concatenate((a, b, c), axis=0)

    y_hat_ec = np.transpose(y_hat_ec)
    
    y_hat = []
    
    for i in range(y_hat_ec.shape[0]):
        y_hat.append(np.bincount(y_hat_ec[i]).argmax())

    return y_hat

#==============================================================================
# ###########################################################
#                         Bibliography                               
# The functions in this file took inspiration from: 
# 
#  - (Raschka, 2015)                          
#
# for further detail please refer to the report bibilography                                      
# ###########################################################
#==============================================================================