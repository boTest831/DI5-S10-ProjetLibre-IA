
import numpy as np
import pandas as pd
import AI_Test.Perceptron.Perceptron as perceptron_class
import AI_Test.Perceptron.Adaline as adaline_class
import AI_Test.Perceptron.MLP as MLP_class
from sklearn.linear_model import LogisticRegression

dataframe = pd.read_csv(r"D:\France\S10\Projet Libre\DI5-S10-ProjetLibre-IA\AI_Test\Perceptron\newtest.csv")


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
#这里是对于感知机模型进行训练
ppn = perceptron_class.Perceptron(eta=0.5, n_iter=10)
# ppn.fit(X_std, y)

# logisticRegre = LogisticRegression(solver='liblinear')
# #训练
# logisticRegre.fit(X_train, y)
# print(logisticRegre.coef_)

# adaline = adaline_class.AdalineSGD(eta=0.01,n_iter=20,random_state=1)
# adaline.fit(X_std, y)

nn = MLP_class.neural_network(X_train, y, 2, 784, 2, is_bias=True)
nn.learning()
np.set_printoptions(suppress=True, precision=2)
print(nn.out_softmax)

# error = 0
# for xi, target in zip(X, y):
#     Z = ppn.predict(xi)
#     if not (Z == target):
#         error += 1
#     if (Z == -1):
#         predict = "circle"
#     elif (Z == 1):
#         predict = "square"
#     if (target == -1):
#         shape = "circle"
#     elif (target == 1):
#         shape = "square"
#     print("Shape:", shape, " Predict: ", predict)
# print("Error:", error)

# print(logisticRegre.predict(X))
# print(logisticRegre.score(X, y))