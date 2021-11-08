"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from tabulate import tabulate
from sklearn import tree
import pickle
import statistics


digits = datasets.load_digits()



n_samples = len(digits.images)
data_ = digits.images.reshape((n_samples, -1))



metric_svm = {}
metric_decision_tree = {}

# Split data into 50% train and 50% test subsets


def DecisionTree():

    X_train, X_test, y_train, y_test = train_test_split(data_, digits.target, test_size=0.2, shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    depth = [i for i in range(1,20)]
    data = []

    model_lst = []


    for dp in depth:

        clf = tree.DecisionTreeClassifier(max_depth=dp)
        clf.fit(X_train, y_train)

        predicted = clf.predict(X_val)
        f1 = round(f1_score(y_val, predicted, average='weighted'),2)
        acc_val = round(accuracy_score(y_val , predicted),2)



        if(acc_val>0.25):
            #print("Storing metrics for depth " ,dp)
            model = [clf ,f1,acc_val]
            model_lst.append(model)

            data.append([dp,f1,acc_val])
        #else:
            #print("Skipping for depth",dp)



    #print(tabulate(data, headers=["Depth","F1-Score(weighted)", "Accuracy Val"]))

    max_a = 0
    idx = 0

    #print(data)

    for i in range(len(data)):
        if(data[i][2] > max_a):
            idx = i
            max_a = data[i][1]

    clf = model_lst[idx][0]


    predicted = clf.predict(X_test)
    acc_test= round(accuracy_score(y_test , predicted),2)
    f1 = round(f1_score(y_test, predicted, average='weighted'),2)

    #print("Best Depth Value : ",data[idx][0])
    #print("F1 score for best Depth ",f1)
    #print("Train accuracy for best Depth ",acc_test)

    best_depth_dt = data[idx][0]
    best_acc_dt = acc_test
    best_f1_dt = f1 

    return best_depth_dt , best_acc_dt , best_f1_dt


def SVM():


    X_train, X_test, y_train, y_test = train_test_split(data_, digits.target, test_size=0.2, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    gamma = [10**i for i in range(-7,7)]
    data = []

    model_lst = []


    for gm in gamma:

        clf = svm.SVC(gamma=gm)
        clf.fit(X_train, y_train)

        predicted = clf.predict(X_val)
        f1 = round(f1_score(y_val, predicted, average='weighted'),2)
        acc_val = round(accuracy_score(y_val , predicted),2)



        if(acc_val>0.25):
            #print("Storing metrics for gamma " ,gm)
            model = [clf ,f1,acc_val]
            model_lst.append(model)

            data.append([gm,f1,acc_val])
        #else:
            #print("Skipping for gamma",gm)



    #print(tabulate(data, headers=["Gamma","F1-Score(weighted)", "Accuracy Val"]))

    max_a = 0
    idx = 0

    #print(data)

    for i in range(len(data)):
        if(data[i][2] > max_a):
            idx = i
            max_a = data[i][1]

    clf = model_lst[idx][0]


    predicted = clf.predict(X_test)
    acc_test = round(accuracy_score(y_test , predicted),2)
    f1 = round(f1_score(y_test, predicted, average='weighted'),2)
    #print("Best Gamma Value : ",data[idx][0])
   # print("Train accuracy for best gamma ",acc_test)
   # print("F1 score for best gamma ",f1)
    return  data[idx][0],acc_test,f1 


metric= []
dt_acc = []
svm_acc = []
dt_f1 = []
svm_f1 = []

for i in range(5):
    print('     Iteration ..............', i+1)
    d,a1,fs1 = DecisionTree()
    b,a2,fs2 = SVM()

    dt_acc.append(a1)
    svm_acc.append(a2)

    dt_f1.append(fs1)
    svm_f1.append(fs2)

    metric.append(['Iteration '+str(i) , b, a2,fs2 ,d,a1,fs1])
    #print(['Iteration '+str(i) , b, a2,fs2 ,d,a1,fs1])
metric.append(['Mean', '--', statistics.mean(svm_acc) , statistics.mean(svm_f1 ),'--',statistics.mean(dt_acc) , statistics.mean(dt_f1 )])
metric.append(['Variance', '--' ,statistics.variance(svm_acc) , statistics.variance(svm_f1 ),'--',statistics.variance(dt_acc) , statistics.variance(dt_f1 )])

print()
print('         ----Summary----')
print()

print(tabulate(metric, headers=[" ","Gamma","SVM Accuracy" ,"SVM F1 Score", "Depth","DT Accuracy" ,"DT F1 Score"]))