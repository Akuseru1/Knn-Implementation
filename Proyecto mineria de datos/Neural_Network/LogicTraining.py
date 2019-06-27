import random
import math

def initValues():
    inpWeights = []
    midWeights = []
    bias = []
    ## se debe poner un numero de inputs
    numInputs = 3
    rand = lambda: random.uniform(-1, 1)
    for i in range(numInputs):
        inpWeights.append([rand() for o in range(numInputs - 1)])
    for j in range(len(inpWeights) - 1):
        midWeights.append(rand())
    for h in range(len(midWeights) + 1):
        bias.append(rand())
    return inpWeights, midWeights, bias


def netOutput(inputs):
    newInput = -1 * inputs
    equation = 1 / (1 + math.exp(newInput))
    return equation


def netInput(inpWeights, midWeights, bias, tupla):
    inputs = 0
    outputs = []
    for w in range(len(inpWeights[0])):
        for unit in range(len(inpWeights)):
            #print("Inpweight [U",unit,"][W",w, "]:",inpWeights[unit][w])
            #print("bias", bias[w])
            inputs += inpWeights[unit][w] * tupla[unit] # solo se calcula el input de 4 y 5
        inputs += bias[w]
        outputs.append(netOutput(inputs)) # 6 queda en la primera pos
        inputs = 0
    finalOutput = 0
    for i in range(len(outputs)):
        finalOutput += midWeights[i] * outputs[i]
    finalOutput += bias[(len(bias) - 1)]
    finalOutput = netOutput(finalOutput)
    outputs.append(finalOutput)
    return outputs


def calcError(outputError, outputs, midWeights):
    errors = []
    for i in range(len(outputs)-1):
        calc = outputs[i] * (1 - outputs[i]) * outputError * midWeights[i]
        errors.append(calc) # 5 queda en ultima posicion
    return errors