from Neural_Network.Training import forwardPropagation,backPropagation,errorNodes, getValues
from sklearn.datasets import load_iris
iris = load_iris()
flower = list(iris.data)
iris2 = load_iris()
flowers = []
for k in iris.data: # llena flowers con los elementos de iris.data
    temp = list(k)
    flowers.append(temp)
def train(infoFlores): # info flores es una tupla con los datos de la flor
    label = iris.target[flowers.index(infoFlores)]
    if label == 0:
        clabel = [1, 0, 0]
    elif label == 1:
        clabel = [0, 1, 0]
    else:
        clabel = [0, 0, 1]
    outputs = forwardPropagation(clabel, infoFlores)
    errors = errorNodes(outputs, clabel)
    backPropagation(errors, outputs) # Mando los calculos de errores y los output

def predict(infoFlores):  # Funcion cuyo objetivo es predecir usando el modelo entrenado
    label = iris.target[flowers.index(infoFlores)]
    if label == 0:
        clabel = [1, 0, 0]
    elif label == 1:
        clabel = [0, 1, 0]
    else:
        clabel = [0, 0, 1]

    output = forwardPropagation(clabel, infoFlores)

    print("La clase es:", label)

    print("Los output son:", output[-3], output[-2], output[-1])
