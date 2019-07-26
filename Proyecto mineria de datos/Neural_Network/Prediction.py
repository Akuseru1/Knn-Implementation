from Neural_Network.Show_Neural_Network import *
#from Neural_Network.Training import getValues


def trainData(iris, flowers, training):     #Entrena los datos
    for iter_entrenamiento in range(100):
        for i in range(len(training)):
            label = iris.target[flowers.index(training[i])]
            train(training[i], label)

def trainDataCV(iris, flowers, training): # mismo que train data pero para cross validation
    for iter_entrenamiento in range(100):
        for i in range(len(training)):
            for j in range(len(training[0])):
                label = iris.target[flowers.index(training[i][j])]
                train(training[i][j], label)


def predictPorcentual(test, training, iris): # predice todas las clases de test
    flowers = []
    for k in iris.data:  # llena flowers con los elementos de iris.data
        temp = list(k)
        flowers.append(temp)
    match = []
    trainData(iris, flowers, training) # entrena

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

    trainDataCV(iris, flowers, training)

    porcent = []
    totalPorcentage = []
    numDiv = len(training) + 1

    for i in range(len(training)):
        for j in range(len(training)):
            match = predictPorcentual(test, training[j], iris)
            porcent.append(match.count(True) / len(match))
        totalPorcentage.append(sum(porcent) / (len(training)))
        porcent = []
        temp = training.pop(0)
        training.append(test)
        test = temp
    hey = int((sum(totalPorcentage) / numDiv) * 100)
    return int((sum(totalPorcentage) / numDiv) * 100)



