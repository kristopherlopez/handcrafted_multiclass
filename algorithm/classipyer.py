#==============================================================================
# ###########################################################
#                        classiPYer                             
#  The following file is used to initialise and direct the model 
#  process. It is the main function for the developed program
# 
#  The user has four options:
#  - Quick training
#  - Long training
#  - Quick testing
#  - Long testing
#
#  Training options are
#   - trained on a 90% of the training data
#   - tested on remaining 10%
#
#  Testing options are
#   - trained on 100% of the training data
#   - tested on test data with predicted_labels output
#
#  Quick options are logistic regression models
#   - 0.55 l2 regularization and all features used
#   - takes approximately 20 minutes to run on 8GB memory machines
#   - takes approximately 10 minutes to run on 16GB memory machines
#   - accuracy approximately 65.2%
#
#  Long options are ensemble models
#   - 4 votes 0.55 l2 logistic regression on 90% retained variance SVD
#   - 2 votes naive bayes with 0% prior weight
#   - 3 votes 20 nearest neighours on 40% retained variance SVD
#   - takes approximately 2 hours to run on 8GB memory machines
#   - takes approximately 30 minutes to run on 16GB memory machines
#   - accuracy approximately 65.8%
#
# ###########################################################
#==============================================================================

import sys, os

file_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(file_directory)
input_directory = os.path.join(parent_directory, 'input/')
output_directory = os.path.join(parent_directory, 'output/')

paths = {}

paths['training_data'] = os.path.join(input_directory, 'training_data.csv').replace('\\', '/')
paths['training_labels'] = os.path.join(input_directory, 'training_labels.csv').replace('\\', '/')
paths['test_data'] = os.path.join(input_directory, 'test_data.csv').replace('\\', '/')
paths['predicted_labels'] = os.path.join(output_directory, 'predicted_labels.csv').replace('\\', '/')

sys.path.append(file_directory)

from evaluation import output
from time import time

startup_text = ('\nWelcome to Group 34\'s classiPYer \n'
+'\n We have a range of options to run our code:'
+'\n------------------------------------------------------------------------------'
+'\n|Option|        Model   |  Classifier | ~Accuracy | Timing 8GB | Timing 16GB |'
+'\n| (1)  | Quick training | Logistic    |   0.652   | 20 minutes | 10 minutes  |'  
+'\n| (2)  | Long training  | Ensemble    |   0.658   | 2  hours   | 30 minutes  |'
+'\n| (3)  | Quick testing  | Logistic    |    N/A    | 20 minutes | 10 minutes  |'
+'\n| (4)  | Long testing   | Ensemble    |    N/A    | 2  hours   | 30 minutes  |'
+'\n------------------------------------------------------------------------------'
+'\n\nInput your selection as a number: ')

start = time()

method = int(input(startup_text))

if method == 1:
	print('\n---Running Quick training Model---\n')    
	output.predict_on_training_quick(paths)
elif method == 2:
	print('\n---Running Long training model---\n')
	output.predict_on_training_long(paths)
elif method == 3:
	print('\n---Running Quick testing model---\n')
	output.predict_on_test_quick(paths)
elif method == 4:
	print('\n---Running Long testing model---\n')
	output.predict_on_test_long(paths)
else:
	print("\nInvalid selection\n--Exiting Program---")

end = time() - start
print('Total time to run %0.3fs' %end)
