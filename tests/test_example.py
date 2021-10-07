import sys 
import os
import warnings
from sklearn.utils.random import sample_without_replacement

warnings.filterwarnings("ignore")

base_path = '/home/anurag/Desktop/ML_ops/MNIST_Example/mnist'        
sys.path.append(base_path)
from plot_graph import classification_task
from sklearn import datasets


def test_model_writing():
	digits = datasets.load_digits()
	metric , model_path = classification_task(digits)
	final_file = '../mnist/' + model_path[1:] 
	assert os.path.isdir(final_file)

def test_small_data_overfit_checking():
	digits = datasets.load_digits()
	metrics , _ = classification_task(digits,isTrain=True)
	assert metrics['acc'] > 0.9
	assert metrics['f1'] > 0.9

