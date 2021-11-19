import sys 
import os
import warnings
#from sklearn.utils.random import sample_without_replacement
import math
from joblib import dump, load

import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from tabulate import tabulate
from sklearn import tree
import pickle
import statistics


warnings.filterwarnings("ignore")



digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
target = digits.target

## Made up generic test case for bonus


## For SVM
#
clf = load('SVM.joblib')

def test_digit_correct_0():
	count = 0
	while(1):
		if target[count] == 0:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 0)

def test_digit_correct_1():
	count = 0
	while(1):
		if target[count] == 1:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 1)


def test_digit_correct_2():
	count = 0
	while(1):
		if target[count] == 2:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 2)

def test_digit_correct_3():
	count = 0
	while(1):
		if target[count] == 3:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 3)

def test_digit_correct_4():
	count = 0
	while(1):
		if target[count] == 4:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 4)


def test_digit_correct_5():
	count = 0
	while(1):
		if target[count] == 5:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 5)


def test_digit_correct_6():
	count = 0
	while(1):
		if target[count] == 6:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 6)


def test_digit_correct_7():
	count = 0
	while(1):
		if target[count] == 7:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 7)


def test_digit_correct_8():
	count = 0
	while(1):
		if target[count] == 8:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 8)

def test_digit_correct_9():
	count = 0
	while(1):
		if target[count] == 9:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 9)


clf = load('DT.joblib')

def test_digit_dt_correct_0():
	count = 0
	while(1):
		if target[count] == 0:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 0)

def test_digit_dt_correct_1():
	count = 0
	while(1):
		if target[count] == 1:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 1)


def test_digit_dt_correct_2():
	count = 0
	while(1):
		if target[count] == 2:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 2)

def test_digit_dt_correct_3():
	count = 0
	while(1):
		if target[count] == 3:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 3)

def test_digit_dt_correct_4():
	count = 0
	while(1):
		if target[count] == 4:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 4)


def test_digit_dt_correct_5():
	count = 0
	while(1):
		if target[count] == 5:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 5)


def test_digit_dt_correct_6():
	count = 0
	while(1):
		if target[count] == 6:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 6)


def test_digit_dt_correct_7():
	count = 0
	while(1):
		if target[count] == 7:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 7)


def test_digit_dt_correct_8():
	count = 0
	while(1):
		if target[count] == 8:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 8)

def test_digit_dt_correct_9():
	count = 0
	while(1):
		if target[count] == 9:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 9)
