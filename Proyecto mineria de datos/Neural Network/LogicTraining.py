import random
import math

def initValues():
    inpWeights = []
    midWeights = []
    weightsUnit1 = []
    weightsUnit2 = []
    weightsUnit3 = []
    bias = []
    rand = lambda: random.uniform(-1, 1)
    for i in range(2):
        weightsUnit1.append(rand())
        weightsUnit2.append(rand())
        weightsUnit3.append(rand())

    inpWeights.append(weightsUnit1)
    inpWeights.append(weightsUnit2)
    inpWeights.append(weightsUnit3)

    weights4, weights5 = [rand() for i in range(2)]
    θ4, θ5, θ6 = [rand() for i in range(3)]

    bias.extend([θ4, θ5, θ6])

    midWeights.extend([weights4, weights5])
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
        errors.append(calc) # 5 queda en la primera pos
    return errors