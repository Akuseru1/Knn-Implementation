from Knn.Distance import getDistance
from sklearn.datasets import load_iris
iris = load_iris()
targets = iris.target # llena targets con los tipos (clases) de las flores
flowers = []
for k in iris.data: # llena flowers con los elementos de iris.data
    temp = list(k)
    flowers.append(temp)


def getAccuracy(test, training, largest):
    distances = []
    matches = []
    count = 0
    for i in range(len(test)):
        # print("numero de test: ", len(test), "y esta en la posicion", i)
        for j in range(len(training)):
            distances.append((getDistance(test[i], training[j]), j)) # el indice de training esta guardandose como tupla
        closestIndexes = closestIn(k_Closest(distances, largest), training) # solo los indices de los 3+ cercanos
        testIndex = testIn(test, i) # un unico index
        matches += match(testIndex, closestIndexes)

    for h in range(len(matches)):
        if matches[h]:
            count += 1
    return count  # es el numero de entrenamiento o el total?


def getAccuracy_CV(test, training, largest):
    porcent = []
    porcentajeTotal = 0
    numelemTraining = len(flowers) - len(training[0])
    for k in range(len(training) + 1):
        porcent.append(0)
    for i in range(len(training)):
        for j in range(len(training)):
            porcent[i] += getAccuracy(test, training[j], largest) / numelemTraining  # Es la suma de todos los matches
        temp = training.pop(0)
        training.append(test)
        test = temp
        numDiv = len(training) + 1

    for u in porcent:
        porcentajeTotal += u
    return int((porcentajeTotal / numDiv ) * 100)

##################################################################################################


def getKey(item): # permite tomar el primer elemento de una tupla [(x, y),(z, w)]
    return item[0]


def k_Closest(distances, largest): # obtiene unicamente los k (largest) datos mas cercanos
    closest = []
    distances.sort(key=getKey, reverse=True)
    for i in range(largest):
        closest.append(distances[i])
    return closest


def closestIn(closest, training): # da los indices de los k datos mas cercanos en la base de datos original
    targetIn = []
    for i in range(len(closest)):
        targetIn.append(flowers.index(list(training[closest[i].__getitem__(1)]))) # funcion index da el indice del elemento
    return targetIn


def testIn(test, testIndex): # consigue el indice del dato test  en la base de datos original
    targetIndex = int(flowers.index(list(test[testIndex])))
    return targetIndex


def match(testIndex, trainingIndex): # compara si la clase de test es igual a los k datos mas cercanos ( trainingIndex tendra k elementos con la distancia y el indice)
    matches = []
    test = int(targets[testIndex])
    for i in range(len(trainingIndex)):
        closest = int(targets[trainingIndex[i]])
        if test == closest:
            matches.append(True)
        else:
            matches.append(False)
    return matches


