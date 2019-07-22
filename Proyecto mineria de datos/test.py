from Neural_Network.Show_Neural_Network import *
from Algorithms import createPercentageDistribution
from sklearn.datasets import load_iris
iris = load_iris()
test, training = createPercentageDistribution(iris, 70)
for iter_entrenamiento in range(100):
    for i in range(len(training)):
        train(training[i])

predict(test[0])