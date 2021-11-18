

import os
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from skimage import data, color
import numpy as np
from joblib import dump, load
from utils import preprocess, create_splits, test_
import glob
import sys 

base_path = '/home/anurag/Desktop/ML_ops/MNIST_Example/mnist'        
sys.path.append(base_path)

digits = datasets.load_digits()
n_samples = len(digits.images)



def classification_task(digits, isTrain = False):

    rescale_factors = [1]
    for test_size, valid_size in [(0.15, 0.15)]:
        for rescale_factor in rescale_factors:
            model_candidates = []
            for gamma in [10 ** exponent for exponent in range(-7, 0)]:
                resized_images = preprocess(
                    images=digits.images, rescale_factor=rescale_factor
                )
                resized_images = np.array(resized_images)
                data = resized_images.reshape((n_samples, -1))

                # Create a classifier: a support vector classifier
                clf = svm.SVC(gamma=gamma)
                
                X_train, X_test, X_valid, y_train, y_test, y_valid = create_splits(
                    data, digits.target, test_size, valid_size
                )

                if isTrain:
                    X_test = X_train = X_valid = data
                    y_train = y_valid = y_test = digits.target
                    

                clf.fit(X_train, y_train)
                metrics_valid = test_(clf, X_valid, y_valid)
                
                # we will ensure to throw away some of the models that yield random-like performance.
                if metrics_valid['acc'] < 0.11:
                    #print("Skipping for {}".format(gamma))
                    continue

                candidate = {
                    "acc_valid": metrics_valid['acc'],
                    "f1_valid": metrics_valid['f1'],
                    "gamma": gamma,
                }
                model_candidates.append(candidate)
                if isTrain:
                    output_folder = base_path+"/models/tt_{}_val_{}_rescale_{}_gamma_{}_train".format(
                    test_size, valid_size, rescale_factor, gamma)
                else :
                    output_folder = base_path+"/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                    test_size, valid_size, rescale_factor, gamma)
            
                os.mkdir(output_folder)
                dump(clf, os.path.join(output_folder, "model.joblib"))

            # Predict the value of the digit on the test subset

            max_valid_f1_model_candidate = max(
                model_candidates, key=lambda x: x["f1_valid"]
            )
            best_model_folder = base_path+ "/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, max_valid_f1_model_candidate["gamma"]
            )

            best_model_file = "/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, max_valid_f1_model_candidate["gamma"]
            )
            clf = load(os.path.join(best_model_folder, "model.joblib"))

            metrics = test_(clf, X_test, y_test)
            return metrics,best_model_file


def classification_task1(digits, isTrain = False):

    rescale_factors = [1]
    for test_size, valid_size in [(0.15, 0.15)]:
        for rescale_factor in rescale_factors:
            model_candidates = []
            for gamma in [10 ** exponent for exponent in range(-7, 0)]:
                resized_images = preprocess(
                    images=digits.images, rescale_factor=rescale_factor
                )
                resized_images = np.array(resized_images)
                data = resized_images.reshape((n_samples, -1))

                # Create a classifier: a support vector classifier
                clf = svm.SVC(gamma=gamma)
                
                X_train, X_test, X_valid, y_train, y_test, y_valid = create_splits(
                    data, digits.target, test_size, valid_size
                )

                if isTrain:
                    X_test = X_train = X_valid = data
                    y_train = y_valid = y_test = digits.target
                    

                clf.fit(X_train, y_train)
                metrics_valid = test_(clf, X_valid, y_valid)
                
                # we will ensure to throw away some of the models that yield random-like performance.
                if metrics_valid['acc'] < 0.11:
                    #print("Skipping for {}".format(gamma))
                    continue

                candidate = {
                    "acc_valid": metrics_valid['acc'],
                    "f1_valid": metrics_valid['f1'],
                    "gamma": gamma,
                }
                model_candidates.append(candidate)
                if isTrain:
                    output_folder = base_path+"/models/tt_{}_val_{}_rescale_{}_gamma_{}_train".format(
                    test_size, valid_size, rescale_factor, gamma)
                else :
                    output_folder = base_path+"/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                    test_size, valid_size, rescale_factor, gamma)
            
                os.mkdir(output_folder)
                dump(clf, os.path.join(output_folder, "model.joblib"))

            # Predict the value of the digit on the test subset

            max_valid_f1_model_candidate = max(
                model_candidates, key=lambda x: x["f1_valid"]
            )
            best_model_folder = base_path+ "/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, max_valid_f1_model_candidate["gamma"]
            )

            best_model_file = "/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, max_valid_f1_model_candidate["gamma"]
            )
            clf = load(os.path.join(best_model_folder, "model.joblib"))

            metrics = test_(clf, X_test, y_test)
            return metrics,best_model_file

i,j = classification_task1(digits)
print(i,j)