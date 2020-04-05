import numpy as np
import pandas as pd
import MLP as MLP_class
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataframe = pd.read_csv('newtest.csv')

y = dataframe.iloc[0:200, 0].values
y = np.where(y == 0, 0, 1)
X = dataframe.iloc[0:200, 1:785].values
for i in range (200):
    for j in range(784):
        X[i][j] = float(X[i][j])

X_std=np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean()) / X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean()) / X[:,1].std()
X_train = np.array(X_std)
data_train, data_test, target_train, target_test = train_test_split(X_train,y,test_size=0.2)

nn = MLP_class.neural_network(data_train, target_train, 2, 784, 2,100, 784, 0, 0, is_bias=True)
# nn = MLP_class.neural_network(X_train, y, 2, 784, 2, is_bias=True)
nn.learning()
np.set_printoptions(suppress=True, precision=2)

print(nn.out_softmax)

count_train_error = 0
for data,label in zip(data_train,target_train):
    print("trainset")
    predict = nn.predict(data).tolist()
    if not label == round(predict[1][0]):
        count_train_error+=1
    if (label == 0):
        label = "circle"
    elif(label == 1):
        label == "square"
    print("label: ",label, "predict", round(predict[1][0]))


count_test_error = 0
for data,label in zip(data_test,target_test):
    print("testset")
    predict = nn.predict(data).tolist()
    if not label == round(predict[1][0]):
        count_test_error+=1
    if(label == 0):
        label = "circle"
    elif(label == 1):
        label == "square"
    print("label: ",label, "predict: ", round(predict[1][0]))

print(count_train_error)
print(count_test_error)

plt.figure()
x = range(0,nn.num_epoch)
plt.plot(x,nn.array_loss, label='loss')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=3, mode="expand", borderaxespad=0.)
plt.show()

print(nn.weights)
# print(logisticRegre.predict(X))
# print(logisticRegre.score(X, y))