import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from tabulate import tabulate
from sklearn import tree
import pickle
import statistics
from joblib import dump, load
import matplotlib.pyplot as plt


digits = datasets.load_digits()



n_samples = len(digits.images)
data_ = digits.images.reshape((n_samples, -1))

X_train_f, X_test, y_train_f, y_test = train_test_split(data_, digits.target, test_size=0.20 ,shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train_f, y_train_f, test_size=0.10, shuffle=True)


metric_decision_tree = {}

def SVM(split):

    gamma = [10**i for i in range(-7,7)]

    if split != 1:
        X_train, _, y_train, _ = train_test_split(X_train_f, y_train_f, test_size=1-split ,shuffle=True)
    else :
        X_train, y_train = X_train_f, y_train_f


    f1_score_list = []


    for gm in gamma:

        clf = svm.SVC(gamma=gm)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_val)
        f1 = round(f1_score(y_val, predicted, average='macro'),2)
        f1_score_list.append(f1)

    return max(f1_score_list)



def DecisionTree(split):

    if split != 1:
        X_train, _, y_train, _ = train_test_split(X_train_f, y_train_f, test_size=1-split ,shuffle=True)
    else :
        X_train, y_train = X_train_f, y_train_f


    depth = [i for i in range(1,21)]
    data = []

    model_lst = []

    f1_score_list = []


    for dp in depth:

        clf = tree.DecisionTreeClassifier(max_depth=dp)
        clf.fit(X_train, y_train)

        predicted_val = clf.predict(X_val)
        acc_val = round(accuracy_score(y_val , predicted_val),2)

        predicted_train = clf.predict(X_train)
        acc_train = round(accuracy_score(y_train , predicted_train),2)

        predicted_test = clf.predict(X_test)
        acc_test = round(accuracy_score(y_test , predicted_test),2)

        f1 = f1_score(y_test, predicted_test , average='macro')
        f1_score_list.append(f1)

        model = [clf,acc_val]
        model_lst.append(model)
        data.append([dp,acc_train,acc_val,acc_test,f1])

    return max(f1_score_list)



train_split = [10,20,30,40,50,60,70,80,90,100]


f1_value_SVM = []
f1_value_DT = []

for i in train_split:
    f1_value_SVM.append(round(SVM(i/100),2))
    f1_value_DT.append(round(DecisionTree(i/100),2))



plt.plot(train_split , f1_value_SVM , color='b',label='SVM Classifier')
plt.plot(train_split , f1_value_DT , color='r',label='Decision Tree Classifier')
plt.xlabel('Train Split')
plt.ylabel('F1 Value')
plt.legend()
plt.title('F1 Score V/s Train Split Plot')
plt.show()
