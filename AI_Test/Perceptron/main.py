
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import AI_Test.Perceptron.Perceptron as perceptron_class

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
dataframe = pd.read_csv(r"D:\France\S10\Projet Libre\DI5-S10-ProjetLibre-IA\AI_Test\Perceptron\newtest.csv")

#抽取出前100条样本，这正好是Setosa和Versicolor对应的样本，我们将Versicolor对应的数据作为类别1，Setosa对应的作为-1。
# 对于特征，我们抽取出sepal length和petal length两维度特征，然后用散点图对数据进行可视化

y = dataframe.iloc[0:200, 0].values
y = np.where(y == 0, -1, 1)
X = dataframe.iloc[0:200, 1:785].values
print(X[0][0])
for i in range (200):
    for j in range(784):
        X[i][j] = float(X[i][j])

# y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', -1, 1)
# X = df.iloc[0:100, [0, 2]].values

# image = Image.open("D:/France/S10/Projet Libre/DI5-S10-ProjetLibre-IA/AI_Test/kaggleData/basicshapes/circles/drawing(1).png") # 用PIL中的Image.open打开图像
# image = image.convert('L')
# image_arr = np.array(image) # 转化成numpy数组
# images=[]
# images.append(image_arr)
# images = np.array(images)
# label =[1]
#这里是对于感知机模型进行训练
ppn = perceptron_class.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

Z = ppn.predict(X[10])
print(y[10])
print(Z)