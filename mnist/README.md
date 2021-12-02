# Assignment 11
### Observation
- As Trainig Sample size increases model is moving more towards generalization. 
- F1 is the harmonic mean of Precision and Recall and gives a better measure of the incorrectly classified cases than the Accuracy Metric.
- Under Small Sample model tends to overfit
- As we are increasing training size we can infer from confuaion matrix that Value at diagonal is getting increases. 
- It means model is able to classify correctly and become less confusive between classes as values at other position decreases as training size increases.
- Confusion matics is presented below.
### Result
![alt text](https://github.com/anurag-saraswat/MNIST_Example/blob/Assignment_11/mnist/Figure_1.png)

## Confusion Matrix

confusion_matrix for taking 10.0 percent training data
[[32  0  0  0  0  1  0  0  0  0]
 [ 0 28  0  0  0  1  0  0  7  2]
 [ 0  0 29  0  0  0  0  1  2  0]
 [ 0  0  0 26  0  0  0  1  3  0]
 [ 0  0  0  0 36  0  0  0  1  0]
 [ 0  0  0  0  1 34  1  0  0  3]
 [ 0  0  0  0  0  0 31  0  0  0]
 [ 0  0  0  0  0  0  0 34  1  3]
 [ 0  0  1  0  0  1  0  0 40  2]
 [ 0  0  0  0  0  0  0  0  1 37]]

confusion_matrix for taking 20.0 percent training data
[[33  0  0  0  0  0  0  0  0  0]
 [ 0 35  0  0  0  0  0  0  0  3]
 [ 0  2 30  0  0  0  0  0  0  0]
 [ 0  0  0 25  0  0  0  2  2  1]
 [ 0  0  0  0 36  0  0  1  0  0]
 [ 0  0  0  0  1 37  0  0  0  1]
 [ 0  1  0  0  0  0 30  0  0  0]
 [ 0  0  0  0  0  0  0 37  0  1]
 [ 0  4  0  0  0  2  0  0 35  3]
 [ 0  1  0  0  1  2  0  0  0 34]]

confusion_matrix for taking 30.0 percent training data
[[33  0  0  0  0  0  0  0  0  0]
 [ 0 38  0  0  0  0  0  0  0  0]
 [ 0  0 32  0  0  0  0  0  0  0]
 [ 0  0  0 28  0  0  0  1  1  0]
 [ 0  0  0  0 37  0  0  0  0  0]
 [ 0  0  0  0  0 38  1  0  0  0]
 [ 0  1  0  0  0  0 30  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  1  0  1  0  0  0  0 42  0]
 [ 0  1  0  0  0  0  0  0  0 37]]

confusion_matrix for taking 40.0 percent training data
[[33  0  0  0  0  0  0  0  0  0]
 [ 0 38  0  0  0  0  0  0  0  0]
 [ 0  0 31  0  0  0  0  1  0  0]
 [ 0  0  0 28  0  0  0  1  1  0]
 [ 0  0  0  0 36  0  0  0  1  0]
 [ 0  0  0  0  0 39  0  0  0  0]
 [ 0  0  0  0  0  0 31  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  1  0  0  0  0  0  0 43  0]
 [ 0  0  0  0  0  0  0  0  0 38]]

confusion_matrix for taking 50.0 percent training data
[[33  0  0  0  0  0  0  0  0  0]
 [ 0 38  0  0  0  0  0  0  0  0]
 [ 0  0 32  0  0  0  0  0  0  0]
 [ 0  0  0 26  0  1  0  1  2  0]
 [ 0  0  0  0 37  0  0  0  0  0]
 [ 0  0  0  0  1 37  1  0  0  0]
 [ 0  0  0  0  0  0 31  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  0  0  0  0  0  0  0 44  0]
 [ 0  0  0  0  0  0  0  0  0 38]]

confusion_matrix for taking 60.0 percent training data
[[33  0  0  0  0  0  0  0  0  0]
 [ 0 38  0  0  0  0  0  0  0  0]
 [ 0  0 32  0  0  0  0  0  0  0]
 [ 0  0  0 28  0  0  0  1  1  0]
 [ 0  0  0  0 36  0  0  0  0  1]
 [ 0  0  0  0  1 37  1  0  0  0]
 [ 0  0  0  0  0  0 31  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  1  0  0  0  0  0  0 43  0]
 [ 0  0  0  0  0  0  0  0  0 38]]

confusion_matrix for taking 70.0 percent training data
[[33  0  0  0  0  0  0  0  0  0]
 [ 0 38  0  0  0  0  0  0  0  0]
 [ 0  0 32  0  0  0  0  0  0  0]
 [ 0  0  0 27  0  1  0  1  1  0]
 [ 0  0  0  0 37  0  0  0  0  0]
 [ 0  0  0  0  0 38  1  0  0  0]
 [ 0  0  0  0  0  0 31  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  0  0  0  0  0  0  0 44  0]
 [ 0  0  0  0  0  0  0  0  0 38]]

confusion_matrix for taking 80.0 percent training data
[[33  0  0  0  0  0  0  0  0  0]
 [ 0 38  0  0  0  0  0  0  0  0]
 [ 0  0 32  0  0  0  0  0  0  0]
 [ 0  0  1 27  0  1  0  1  0  0]
 [ 0  0  0  0 37  0  0  0  0  0]
 [ 0  0  0  0  0 38  1  0  0  0]
 [ 0  0  0  0  0  0 31  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  0  0  0  0  0  0  0 44  0]
 [ 0  0  0  0  0  0  0  0  0 38]]

confusion_matrix for taking 90.0 percent training data
[[33  0  0  0  0  0  0  0  0  0]
 [ 0 38  0  0  0  0  0  0  0  0]
 [ 0  0 32  0  0  0  0  0  0  0]
 [ 0  0  0 28  0  1  0  1  0  0]
 [ 0  0  0  0 36  0  0  1  0  0]
 [ 0  0  0  0  1 37  1  0  0  0]
 [ 0  0  0  0  0  0 31  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  0  0  0  0  0  0  0 44  0]
 [ 0  0  0  0  0  0  0  0  0 38]]

confusion_matrix for taking 100.0 percent training data
[[33  0  0  0  0  0  0  0  0  0]
 [ 0 38  0  0  0  0  0  0  0  0]
 [ 0  0 32  0  0  0  0  0  0  0]
 [ 0  0  0 28  0  1  0  1  0  0]
 [ 0  0  0  0 37  0  0  0  0  0]
 [ 0  0  0  0  0 38  1  0  0  0]
 [ 0  0  0  0  0  0 31  0  0  0]
 [ 0  0  0  0  0  0  0 38  0  0]
 [ 0  0  0  0  0  0  0  0 44  0]
 [ 0  0  0  0  0  0  0  0  0 38]]
