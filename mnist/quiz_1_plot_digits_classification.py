"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.metrics import accuracy_score
import numpy as np

# Data Loading and prepration

digits = datasets.load_digits()
    
n_samples = len(digits.images)
data = digits.images
target = digits.target

data_8_8 = data
data_32_32 = []
data_64_64 = []

print("Shape of data 8x8:",data.shape)

for i in range(data.shape[0]):
  data_32_32.append(resize(data[i], (32, 32)))
data_32_32 = np.array(data_32_32)
print("Shape of data 32_32:",data_32_32.shape)

for i in range(data.shape[0]):
  data_64_64.append(resize(data[i], (64, 64)))
data_64_64 = np.array(data_64_64)
print("Shape of data 64_64:",data_64_64.shape)
 

def main_function(x,y,split_size):

    x = x.reshape((len(x), -1))    

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split_size, shuffle=False)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    acc = accuracy_score(y_test , predicted)

    return acc

print("################ Accuracy Table ######################### ")

# 8X8
acc = main_function(data , target,0.1)
print("Accuracy with size 8x8 and 90:10 split" , round(acc,2))

acc = main_function(data , target,0.2)
print("Accuracy with size 8x8 and 80:20 split" , round(acc,2))

acc = main_function(data , target,0.3)
print("Accuracy with size 8x8 and 70:30 split" , round(acc,2))


print("--------------------------------------------------------------")

# 32X32
acc = main_function(data_32_32 , target,0.1)
print("Accuracy with size 32X32 and 90:10 split" , round(acc,2))

acc = main_function(data_32_32 , target,0.2)
print("Accuracy with size 32X32 and 80:20 split" , round(acc,2))

acc = main_function(data_32_32 , target,0.3)
print("Accuracy with size 32X32 and 70:30 split" , round(acc,2))

print("--------------------------------------------------------------")

# 64X64
acc = main_function(data_64_64 , target,0.1)
print("Accuracy with size 64X64 and 90:10 split" , round(acc,2))

acc = main_function(data_64_64 , target,0.2)
print("Accuracy with size 64X64 and 80:20 split" , round(acc,2))

acc = main_function(data_64_64 , target,0.3)
print("Accuracy with size 64X64 and 70:30 split" , round(acc,2))