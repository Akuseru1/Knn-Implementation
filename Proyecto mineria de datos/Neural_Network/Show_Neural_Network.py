from Neural_Network.Training import forwardPropagation,backPropagation,errorNodes
from sklearn.datasets import load_iris
iris = load_iris()
flower = list(iris.data)
iris2 = load_iris()
flowers = []
for k in iris.data: # llena flowers con los elementos de iris.data
    temp = list(k)
    flowers.append(temp)


def train(infoFlores, label): # info flores es una tupla con los datos de la flor
    if label == 0:
        clabel = [1, 0, 0]
    elif label == 1:
        clabel = [0, 1, 0]
    else:
        clabel = [0, 0, 1]
    outputs = forwardPropagation(clabel, infoFlores)
    errors = errorNodes(outputs, clabel)
    backPropagation(errors, outputs) # Mando los calculos de errores y los output


def showClass(infoFlores, label):  # Funcion cuyo objetivo es predecir usando el modelo entrenado
    if label == 0:
        clabel = [1, 0, 0]
    elif label == 1:
        clabel = [0, 1, 0]
    else:
        clabel = [0, 0, 1]
    output = forwardPropagation(clabel, infoFlores) # devuelve todos los outputs: los ultimos 3 son los de las 3 neuronas de salida
    output = [output[-3], output[-2], output[-1]]
    prediction = max(output)
   # print("La clase es:", label)
   # print("La prediccion fue de: ", output.index(prediction))
   # print("Los output son:", output[-3], output[-2], output[-1])
    return output.index(prediction)   # devuelve el indice del valor mas grande, el cual puede ser 0,1, 2
