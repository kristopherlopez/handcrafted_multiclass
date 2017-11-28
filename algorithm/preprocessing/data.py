#==============================================================================
# ###########################################################
#                        Output                             
#  This file contains the functions used for getting data from 
# inputs, splitting data and converting labels
# 
#  + get_training_data: gets data from input and converts alphanumeric
#  labels to numeric
#  + get_test_data: gets test data from input
#  + split_data:
#       - if size is given, splits training data and labels into  
#       two components to be used for testing
#       - if size is not given returns data and labels
#  + convert_labels: converts labels from numeric back to alphanumeric
# 
# ###########################################################
#==============================================================================

import numpy as np
import pandas as pd

from time import time

def get_training_data(data_path, labels_path):
    print('### Get data ###')
    
    #Load training data    
    start = time()
    data = pd.read_csv(data_path, delimiter=',',header=None)
    end = time() - start
    print ("took %0.3fs to load training data" %end)
    
    #Load training data labels
    start = time()
    labels = pd.read_csv(labels_path, delimiter=',',header=None)
    end = time() - start
    print ("took %0.3fs to load labels" %end)
            
    #Merge data
    start = time()    
    data = pd.merge(labels,data,on=[0,0])
    
    #Convert to categorical column and numbered
    data['1_x'] = data['1_x'].astype('category').cat.codes
    data = data.as_matrix(columns=data.columns[1:])  
    end = time() - start    
    print ("took %0.3fs to merge and convert categorical columns" %end)

    return data
    
def get_test_data(test_path):
    print('### Get test data ###')
    
    #Load test data
    start = time()
    data = pd.read_csv(test_path, delimiter=',',header=None)
    end = time() - start
    print ("took %0.3fs to load test data" %end)
    
    X = data.as_matrix(columns=data.columns[1:])
    labels = data.as_matrix(columns=data.columns[:1])
    
    return X, labels

def split_data(data, size = 0):
    print('### Split data ###')
    
    # Check to see if size is valid
    if size < 0 or size > 1:
        print("test set 'size' must be between 0 and 1")
        return
    
    if size == 0:
        return data[:,1:], data[:,0]
        
    rows = int(data.shape[0] * (1 - size))
    
    # Shuffle data
    
    start = time()
    np.random.shuffle(data)
    end = time() - start
    print ("took %0.3fs to shuffle data " %end)
    
    start = time()
    X, y = data[:,1:], data[:,0]
    
  # Remove features with zero tf-idf entries
    zero_list = []
    for i in range(np.shape(X)[1]):
        if sum(X[:,i]) == 0:
            zero_list.append(i)

    X = np.delete(X, zero_list, 1)
    end = time() - start
    print ("took %0.3f s preprocess data" %end)    
    
    return X[:rows,:], X[rows:,:], y[:rows], y[rows:], zero_list
    
def convert_labels(Y):
        
    # Convert labels to categorical column and numbered
    Category2ID = {
        "Arcade and Action": 0,
        "Books and Reference": 1,
        "Brain and Puzzle": 2,
        "Business": 3,
        "Cards and Casino": 4,
        "Casual": 5,
        "Comics": 6,
        "Communication": 7,
        "Education": 8,
        "Entertainment": 9,
        "Finance": 10,
        "Health and Fitness": 11,
        "Libraries and Demo": 12,
        "Lifestyle": 13,
        "Media and Video": 14,
        "Medical": 15,
        "Music and Audio": 16,
        "News and Magazines": 17,
        "Personalization": 18,
        "Photography": 19,
        "Productivity": 20,
        "Racing": 21,
        "Shopping": 22,
        "Social": 23,
        "Sports": 24,
        "Sports Games": 25,
        "Tools": 26,
        "Transportation": 27,
        "Travel and Local": 28,
        "Weather": 29
    }
    
    # for converting back to text
    ID2Category = {y:x for x,y in Category2ID.items()}
    
    # to convert back
    # take any numpy array of predictions
    prediction = pd.DataFrame(Y).replace(ID2Category,regex=False).as_matrix()
    
    return prediction

#==============================================================================
# ###########################################################
#                         Bibliography                               
# The functions in this file took inspiration from: 
# 
#  - (pberkes, 2010)                        
# 
# for further detail please refer to the report bibilography                                      
# ###########################################################
#==============================================================================