import sys 
import os
import warnings
from sklearn.utils.random import sample_without_replacement
import math

warnings.filterwarnings("ignore")

base_path = '/home/anurag/Desktop/ML_ops/MNIST_Example/mnist'        
sys.path.append(base_path)
from plot_graph import classification_task
from sklearn import datasets
from utils import create_splits



## Made up generic test case for bonus

def test_create_split_bonus():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	val_ratio = 0.2
	test_ratio = 0.1
	train_ratio = 1 - val_ratio - test_ratio

	train_sample = math.ceil(n_samples *train_ratio)
	test_sample = math.ceil(n_samples *test_ratio)
	val_sample = math.ceil(n_samples *val_ratio)

	actual_train ,actual_test , actual_valid ,_ ,_ ,_= create_splits(digits.images, digits.target, test_ratio, val_ratio)    
    
	total = len(actual_train) + len(actual_test) + len(actual_valid) 

	assert train_sample == len(actual_train)
	assert test_sample == len(actual_test)
	assert val_sample == len(actual_valid)
	assert n_samples == total


def test_create_split_1():
	digits = datasets.load_digits()
	n_samples = 100
	val_ratio = 0.7
	test_ratio = 0.2
	train_ratio = 1 - val_ratio - test_ratio

	train_sample = int(n_samples *train_ratio)
	test_sample = int(n_samples *test_ratio)
	val_sample = int(n_samples *val_ratio)

	actual_train ,actual_test , actual_valid ,_ ,_ ,_= create_splits(digits.images[:n_samples ], digits.target[:n_samples ], test_ratio, val_ratio)    
    
	total = len(actual_train) + len(actual_test) + len(actual_valid) 

	assert train_sample == len(actual_train)
	assert test_sample == len(actual_test)
	assert val_sample == len(actual_valid)
	assert n_samples == total

def test_create_split_2():
	digits = datasets.load_digits()
	n_samples = 9
	val_ratio = 0.7
	test_ratio = 0.2
	train_ratio = 1 - val_ratio - test_ratio

	train_sample = math.ceil(n_samples *train_ratio)
	test_sample = math.ceil(n_samples *test_ratio)
	val_sample = math.ceil(n_samples *val_ratio)

	actual_train ,actual_test , actual_valid = create_splits(digits.images[:n_samples ], digits.target[:n_samples ], test_ratio, val_ratio,case = True)    
    
	total = actual_train +actual_test + actual_valid 

	assert train_sample == len(actual_train)
	assert test_sample == len(actual_test)
	assert val_sample == len(actual_valid)
	assert n_samples == total



