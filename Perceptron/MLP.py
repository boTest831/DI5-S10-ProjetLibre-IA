import numpy as np
from scipy import sparse
from sklearn import datasets
from sklearn.model_selection import train_test_split


class neural_network(object):
    # Initialisation du perceptron
    def __init__(self, data, target, num_layer, input_size, output_size, num_epoch=100000, lr=0.001, is_bias=False):
        self.num_layer = num_layer  # nombre de couches de réseau (couches cachées + couche sortie)
        self.data = data.T
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())  # Données normalisées
        self.lr = lr
        target = self.convert_labels(target, output_size)  # Résultat idéal
        self.labels = target
        self.is_bias = is_bias
        self.num_epoch = num_epoch  # Nombre d'itérations

        # Pour dessiner la figure
        self.array_loss = []
        weight = []
        nb_neurons_prec = input_size  # Nombre d'entrées = 4
        biases = []

        for i in range(num_layer - 1):
            # Entrez le nombre de neurones dans chaque couche cachée
            nb_neurons = int(input('Number of neurones in layer' + str(i + 1) + '?'))
            # nb_neurons = 784
            # Initialiser les poids et les biais pour chaque couche avec des nombres aléatoires
            weight_i = np.random.randn(nb_neurons_prec,
                                       nb_neurons)  # Nb poids =  nb données d'entrée dans la couche précédente * Nb neurones dans cette couche
            if is_bias:
                bias = np.random.randn(nb_neurons, 1)  # Nb biases =  Nb neurones
                biases.append(bias)
            weight.append(weight_i)
            nb_neurons_prec = nb_neurons
        weight_last = np.random.randn(nb_neurons_prec, output_size)
        weight.append(weight_last)
        self.weights = weight  # Matrice de poids
        self.biases = biases  # Vecteur de biais

    # propager l'entrée à la dernière couche
    def forward(self):
        self.Z = []  # Sortie de nœuds neuronaux
        self.A = []  # Valeur d'entrée
        a = self.data
        self.A.append(a)
        # Calculez la valeur de sortie de chaque nœud
        for i in range(self.num_layer - 1):
            # zl = self.weights[l] * self.data[l-1] + self.biases[l]
            #
            if self.is_bias:
                z = np.dot(self.weights[i].T, a) + self.biases[i]
            else:
                z = np.dot(self.weights[i].T, a)
            a = np.maximum(z, 0)  # relu Fonction d'activation
            self.Z.append(z)
            self.A.append(a)
        out = np.dot(self.weights[self.num_layer - 1].T, a)  # Résultats de la couche de sortie
        self.out_softmax = self.softmax(out)

    # calcul erreur , rétropropagation via le réseau
    def backward(self):
        self.E = []
        self.dW = []
        self.db = []
        # eL =∂J / ∂zL  le gradient de l’erreur selon bL = eL
        e = (self.out_softmax - self.labels) / (self.labels.shape[1])
        dw = np.dot(self.A[-1], e.T)  # Calcul le gradient de l’erreur selon WL
        self.dW.append(dw)

        # calcul d’erreur de sortie de la couche l
        for i in range(self.num_layer - 1):
            a = self.A[-i - 2]
            z = self.Z[-i - 1]
            e = np.dot(self.weights[-i - 1], e)
            e[z <= 0] = 0
            dw = np.dot(a, e.T)
            if self.is_bias:
                db = np.sum(e, axis=1, keepdims=True)
                self.db.append(db)
            self.dW.append(dw)

        # Ajuster les poids et les biais
        for i in range(self.num_layer):
            self.weights[i] += -self.lr * self.dW[-i - 1]
        if self.is_bias:
            for i in range(self.num_layer - 1):
                self.biases[i] += -self.lr * self.db[-i - 1]

    def learning(self):
        for epoch in range(self.num_epoch):
            self.forward()
            self.loss = self.cost(self.labels, self.out_softmax)  # Calculez l'écart
            self.array_loss.append(self.loss)
            print(epoch, self.loss)
            self.backward()

    # Softmax score
    def softmax(self, V):
        e_V = np.exp(V - np.max(V, axis=0, keepdims=True))
        Z = e_V / e_V.sum(axis=0)
        return Z

    def predict(self, data):
        self.A = []  # Valeur d'entrée
        a = data.reshape(data.shape[0],1)
        self.A.append(a)
        # Calculez la valeur de sortie de chaque nœud
        for i in range(self.num_layer - 1):
            # zl = self.weights[l] * self.data[l-1] + self.biases[l]
            if self.is_bias:
                z = np.dot(self.weights[i].T, a) + self.biases[i]
            else:
                z = np.dot(self.weights[i].T, a)
            a = np.maximum(z, 0)  # relu Fonction d'activation
            self.Z.append(z)
            self.A.append(a)
        out = np.dot(self.weights[self.num_layer - 1].T, a)  # Résultats de la couche de sortie
        return self.softmax(out)

    # One-hot coding
    def convert_labels(self, y, C=2):
        Y = sparse.coo_matrix((np.ones_like(y),
                               (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
        return Y

    # cost or loss function
    def cost(self, Y, Yhat):
        return -np.sum(Y * np.log(Yhat)) / Y.shape[1]


