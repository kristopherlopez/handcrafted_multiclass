#==============================================================================
# ###########################################################
#                        Metrics                             
#  The following file outlines the functions used for evalu-
# ating model performance:
# 
#  + confusionMatrix: produces a numpy CM and will store it
# as an excel file in the output folder if given a path
#  + accuracy: calculates the models accuracy
#  + precision: calculates the models precision
#  + recall: calculates the models recall
#  + fscore: calcualtes the models fscore
# 
#  Note: all aggregation across categories is 'macro'
# ###########################################################
#==============================================================================

import numpy as np
import xlsxwriter

def confusionMatrix(expected,predicted,ModelType,paths=None,numClass=None):
    
    #check if given number of classes, else work it out
    if numClass is None:
        numClass = len(np.unique(expected))
        #print("Number of classes assumed to be {}".format(numClass))
        
    #create an empty matrix of 0s
    cMatrix = [[0] * numClass for i in range(numClass)]
    
    #go through each line and increase its location in the confusion matrix
    for predLine, expLine in zip(predicted, expected):
        cMatrix[int(expLine)][int(predLine)] += 1
    cMatrix = np.asarray(cMatrix)
    
    #write to excel if given a path directory
    if paths is not None:
        results_path = paths['predicted_labels']
        workbook = xlsxwriter.Workbook(results_path.replace('predicted_labels.csv',ModelType+'_cMatrix.xlsx'))
        worksheet = workbook.add_worksheet()
        row = 0
        for col, data in enumerate(cMatrix):
            worksheet.write_column(row, col, data)
        workbook.close()
    
    return cMatrix
    
def accuracy(cMatrix):
    #TP/ALL
    t = cMatrix.sum()
    return sum(cMatrix[i,i] for i in range(len(cMatrix))) / float(t)
    
def precision(cMatrix):
    #TP/(TP+FP)
    precisions=[]
    for i in range(len(cMatrix)):
        TP = cMatrix[i,i]
        FP = sum(cMatrix[:,i])-TP
        precisions.append(TP / float(TP + FP))

    return sum(precisions)/len(cMatrix)
    
def recall(cMatrix):
    #TP/TP+FN
    recall=[]
    for i in range(len(cMatrix)):
        TP = cMatrix[i,i]
        FN = sum(cMatrix[i,:])-TP
        recall.append(TP / float(TP + FN))

    return sum(recall)/len(cMatrix)
    
def fscore(cMatrix):
    #2TP/(2TP + FP + FN)
    fscore=[]
    for i in range(len(cMatrix)):
        TP = cMatrix[i,i]
        FN = sum(cMatrix[i,:])-TP
        FP = sum(cMatrix[:,i])-TP
        fscore.append((2*TP) / float(2*TP + FN + FP))

    return sum(fscore)/len(cMatrix)
    
#==============================================================================
# ###########################################################
#                         Bibliography                               
# The functions in this file took inspiration from: 
# 
#  - (Boltzmann, 2010)                              
#  - (Yang, 2014)                              
# 
# for further detail please refer to the report bibilography                                      
# ###########################################################
#==============================================================================
