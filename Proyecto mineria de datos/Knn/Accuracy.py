from Knn.Distance import getDistance
from sklearn.datasets import load_iris
iris = load_iris()
targets = iris.target # llena targets con los tipos (clases) de las flores
flowers = []
for k in iris.data: # llena flowers con los elementos de iris.data
    temp = list(k)
    flowers.append(temp)


def getAccuracy(test, training, largest):  #Obtiene la presicion en Knn
    distances = []
    matches = []
    count = 0
    for i in range(len(test)):
        # print("numero de test: ", len(test), "y esta en la posicion", i)
        for j in range(len(training)):
            distances.append((getDistance(test[i], training[j]), j)) # el indice de training esta guardandose como tupla
        closestIndexes = closestIn(k_Closest(distances, largest), training) # solo los indices de los 3+ cercanos (knn)
        testIndex = testIn(test, i) # un unico index
        matches.append(match(testIndex, closestIndexes)) # esto es el clasificador, no deberia ir en knn # page 80 principles, 225
    return matches.count(True)


def getAccuracy_CV(test, training, largest): # Obtiene la presicion en validacion crusada
    porcent = []
    totalPorcentage = []
    numDiv = len(training) + 1
    for i in range(len(training)):
        for j in range(len(training)):
            porcent.append(getAccuracy(test, training[j], largest))
        totalPorcentage.append(sum(porcent) / (len(training) * numDiv))
        porcent = []
        temp = training.pop(0)
        training.append(test)
        test = temp
    return int((sum(totalPorcentage) / numDiv ) * 100)

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
    closest = 0
    train = []
    for j in range(len(trainingIndex)):
        train.append(targets[trainingIndex[j]])
   # print("El label real es: ",test)
    #print("Los mas cercanos son: ", train)
    for i in range(len(trainingIndex)):
        if train.count(train[i]) >= 2:
            if(train[i] == test):
                #print("El mas comun:", train[i], "es igual a ", test)
                return True
            else:
                #print("El mas comun:", train[i], "no es igual a ", test)
                return False

