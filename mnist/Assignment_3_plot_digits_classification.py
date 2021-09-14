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
from sklearn.metrics import f1_score,accuracy_score
from tabulate import tabulate

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()



###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))



# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

data = []
gamma = [10**i for i in range(-7,3)]



for gm in gamma:

    clf = svm.SVC(gamma=gm)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 = round(f1_score(y_test, predicted, average='weighted'),2)
    acc = round(accuracy_score(y_test , predicted),2)

    data.append([gm,f1,acc])


print(tabulate(data, headers=["Gamma", "Accuracy", "F1-Score(weighted)"]))




#plt.show()
