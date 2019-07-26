from Neural_Network.Show_Neural_Network import *
#from Neural_Network.Training import getValues


def trainData(iris, flowers, training):
    for iter_entrenamiento in range(100):
        for i in range(len(training)):
            label = iris.target[flowers.index(training[i])]
            train(training[i], label)


def predictPorcentual(test, training, iris):
    flowers = []
    for k in iris.data:  # llena flowers con los elementos de iris.data
        temp = list(k)
        flowers.append(temp)
    match = []
    trainData(iris, flowers, training)

    for i in range(len(test)):
        label = iris.target[flowers.index(test[i])]
       # print("Antes de predecir: inputW, midW, bias")
        #getValues()
        clase = showClass(test[i], label)
        if clase == label:   # si la clase devuelta es igual a la clase registrada, se ingresa un true a match
            match.append(True)
        else:
            match.append(False)
   # getValues()
    return match


def predictCV(test, training, iris):
    flowers = []
    for k in iris.data:  # llena flowers con los elementos de iris.data
        temp = list(k)
        flowers.append(temp)
    match = []
    trainData(iris, flowers, training)
    for i in range(len(test)):
        label = iris.target[flowers.index(test[i])]
        clase = showClass(test[i], label)
        if clase == label:   # si la clase devuelta es igual a la clase registrada, se ingresa un true a match
            match.append(True)
        else:
            match.append(False)
    return match
