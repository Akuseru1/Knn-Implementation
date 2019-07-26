import random
import math

def initValues(numInputs):
    inpWeights = []
    midWeights = []
    bias = []
    ## se debe poner un numero de inputs
    numTargets = 3                         # numero de targets ( 3 tipos de flores)
    numNeurons = numInputs + 2 # Puede variar
    rand = lambda: random.uniform(-1, 1)
    for i in range(numInputs):   # number of inputs
        inpWeights.append([rand() for o in range(numNeurons)])
    for j in range(numNeurons):  # number of neurons in Hidden Layer
        midWeights.append([rand() for i in range(numTargets)])
    for h in range(len(midWeights) + numTargets):  # number of bias values
        bias.append(rand())
    return inpWeights, midWeights, bias


def netOutput(inputs):  # Activa las neuronas
    newInput = -1 * inputs
    equation = 1 / (1 + math.exp(newInput))
    return equation


def netInput(inpWeights, midWeights, bias, tupla, clabel):
    inputs = 0
    outputs = []
    for w in range(len(inpWeights[0])):
        for unit in range(len(inpWeights)):
            inputs += inpWeights[unit][w] * tupla[unit]
        inputs += bias[w]
        outputs.append(netOutput(inputs))
        inputs = 0
    finalOutputs = 0
    for errorNode in range(len(clabel)):
        for unitM in range(len(midWeights)):
            finalOutputs = finalOutputs + midWeights[unitM][errorNode] * outputs[unitM]
        position = finalOutputs + bias[len(bias) - (3 - errorNode)]  # desde el tercer ultimo al ultimo
        outputs.append(netOutput(position))
        finalOutputs = 0
    return outputs  # 9 outputs!


def calcError(outputError, outputs, midWeights, clabel, nodo): #returns errors for 1 neuron in the last layer
    errors = []
    for unit in range(len(outputs) - len(clabel)):
        errors.append(outputs[unit] * (1 - outputs[unit]) * outputError * midWeights[unit][nodo])# i no debe cambiar
    return errors