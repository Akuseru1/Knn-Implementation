from Neural_Network.LogicTraining import initValues, netInput, calcError
tupla = (1, 0, 1, 1)
trainingRate = 0.9
inpWeights, midWeights, bias = initValues(len(tupla))
# inpWeights = [[0.2, -0.3],[0.4, 0.1], [-0.5, 0.2]] # los ejemplos del libro
# midWeights = [-0.3, -0.2]
# bias = [-0.4, 0.2, 0.1]

def forwardPropagation(clabel, datosFlores):
    tupla = datosFlores
    outputs = netInput(inpWeights, midWeights, bias, tupla,clabel)
    return outputs # 9 outputs


def errorNodes(outputs, clabel): # envia a errors (una lista) los errores y añadimos el ultimo explicitamente
    lastlayer = []
    errorsEachNeuron = []
    errors = []
    for i in range(len(clabel)):
        lastlayer.append(outputs[len(outputs)-(3 - i)] * (1 - outputs[len(outputs)-(3 - i)]) * (clabel[i] - outputs[len(outputs)-(3 - i)]))
        errorsEachNeuron.append(calcError(lastlayer[i], outputs, midWeights, clabel, i))
    for j in range(len(errorsEachNeuron[0])):  # para el error de un hidden layer se suman todos los errores
        for h in range(len(errorsEachNeuron)): # generados por los errores de las neuronas de salida
            suma = errorsEachNeuron[h][j]
        errors.append(suma)
        suma = 0
    errors.extend(lastlayer)
    return errors  # 21 errores, divididos en 4 listas, la cuarta es de 3 elementos, los nodos de salida


def backPropagation(errors, outputs): # funcion que actualiza todos los valores
    for i in range(len(midWeights[0])):
        for unit in range(len(midWeights)): # Updates hidden layer weights
            midWeights[unit][i] = midWeights[unit][i] + ((trainingRate * errors[unit]) * outputs[unit])
    for unit in range(len(inpWeights)):
        for w in range(len(inpWeights[0])):
            inpWeights[unit][w] = inpWeights[unit][w] + trainingRate * errors[w] * tupla[unit] # Updates input weights
    for biases in range(len(bias)):
        bias[biases] = bias[biases] + trainingRate * errors[biases] #  Updates bias values

def getValues():
    return inpWeights, midWeights, bias