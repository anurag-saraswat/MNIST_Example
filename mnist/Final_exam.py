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


digits = datasets.load_digits()



n_samples = len(digits.images)
data_ = digits.images.reshape((n_samples, -1))



metric_decision_tree = {}



def DecisionTree():

    X_train, X_test, y_train, y_test = train_test_split(data_, digits.target, test_size=0.15 ,shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, shuffle=True)

    depth = [i for i in range(1,21)]
    data = []

    model_lst = []


    for dp in depth:

        clf = tree.DecisionTreeClassifier(max_depth=dp)
        clf.fit(X_train, y_train)

        predicted_val = clf.predict(X_val)
        acc_val = round(accuracy_score(y_val , predicted_val),2)

        predicted_train = clf.predict(X_train)
        acc_train = round(accuracy_score(y_train , predicted_train),2)

        predicted_test = clf.predict(X_test)
        acc_test = round(accuracy_score(y_test , predicted_test),2)



        # Saving Model
        model_name = './mnist/models/'+'model_'+str(dp)+'_'+str(round(acc_val,2))+'.joblib' 
        dump(clf, model_name)



        model = [clf,acc_val]
        model_lst.append(model)
        data.append([dp,acc_train,acc_val,acc_test])

    return data


metric = []

for i in range(3):
    metric.append(DecisionTree())


final = []
final.append([' ', ['train' , 'val' , 'test'] , ['train' , 'val' , 'test'] , ['train' , 'val' , 'test'] , ['train' , 'val' , 'test']])

for i in range(20):
    mean_ = []
    temp1 = []
    temp1.append(i+1)
    

    temp = []
    for j in range(3):
        temp.append(metric[j][i][1])
        #temp1.append(metric[j][i][1])
    temp1.append(temp)
    mean_.append(round(statistics.mean(temp[1:]),2))

    temp = []
    for j in range(3):
        temp.append(metric[j][i][2])
        #temp1.append(metric[j][i][2])
    temp1.append(temp)
    mean_.append(round(statistics.mean(temp[1:]),2))

    temp = []
    for j in range(3):
        temp.append(metric[j][i][3])
        #temp1.append(metric[j][i][3])
    temp1.append(temp)
    mean_.append(round(statistics.mean(temp[1:]),2))
    
    temp1.append(mean_)
    final.append(temp1)
    




print(tabulate(final, headers=["Depth","Run1" ,"Run2" ,"Run3", "Average" ]))

#print(tabulate(final, headers=["Depth","Run1" , " " , " ","Accuracy_Run2" ," " , " ","Accuracy_Run3"," " , " ", "Average" ," " , " "]))









# metric= []
# dt_acc = []
# svm_acc = []
# dt_f1 = []
# svm_f1 = []

# for i in range(5):
#     print('     Iteration ..............', i+1)
#     d,a1,fs1 = DecisionTree()
#     b,a2,fs2 = SVM()

#     dt_acc.append(a1)
#     svm_acc.append(a2)

#     dt_f1.append(fs1)
#     svm_f1.append(fs2)

#     metric.append(['Iteration '+str(i) , b, a2,fs2 ,d,a1,fs1])
#     #print(['Iteration '+str(i) , b, a2,fs2 ,d,a1,fs1])
# metric.append(['Mean', '--', statistics.mean(svm_acc) , statistics.mean(svm_f1 ),'--',statistics.mean(dt_acc) , statistics.mean(dt_f1 )])
# metric.append(['Variance', '--' ,statistics.variance(svm_acc) , statistics.variance(svm_f1 ),'--',statistics.variance(dt_acc) , statistics.variance(dt_f1 )])

# print()
# print('         ----Summary----')
# print()

# print(tabulate(metric, headers=[" ","Gamma","SVM Accuracy" ,"SVM F1 Score", "Depth","DT Accuracy" ,"DT F1 Score"]))