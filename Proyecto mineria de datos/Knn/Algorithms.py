from Knn.Accuracy import getAccuracy, getAccuracy_CV
import random


def algorithm(iris):
    print("Que modelo desea utilizar?")
    choice = 0
    while choice != 1 and choice != 2:
        print("Distribucion 70 / 30  (1)")
        choice = input("Cross-Validation  (2)")
        choice = int(choice)
    if choice == 1:
        percentage = percentageAdd()
        ans = percentageDistribution(iris, percentage)
        print("La precision usando el ", percentage, "es: ", ans)
        return ans
    else:
        ans = cross_Validation(iris)
        print("La precision usando Validacion-Cruzada es: ", ans)
        return ans


def percentageDistribution(iris, percentage):
    test, training = createPercentageDistribution(iris, percentage)
    showData(test,training)
    ans = str(int((getAccuracy(test, training, 3) / len(training)) * 100))+"%"
    return ans


def cross_Validation(iris):
    test, training = crearListas_CV(iris)
    showData(test, training)
    ans = str(getAccuracy_CV(test, training, 3))+"%"
    return ans


#########################################################################
def percentageAdd():
    percentage = 50
    while percentage < 51 or percentage > 99:
        percentage = input("Ingrese el percentage de datos que desea evaluar (datos de entrenamiento)")
        percentage = int(percentage)
        if percentage < 51:
            print("deben de haber mas datos de entramiento que de prueba")
        elif percentage > 99:
            print("no hay nada con que comparar")
    return percentage

def createPercentageDistribution(iris, percentage):
    # Se consigue el percentage de datos que el usuario desea evaluar
    print("numero total de datos: ", len(iris.data))

    numdatosT = percentage / 100
    numdatosT = int(len(iris.data) * numdatosT)

    numdatosP = len(iris.data) - numdatosT
    print("numero de datos de entrenamiento(", percentage, "%): ", numdatosT)
    print("numero de datos de prueba(", 100 - percentage, "%): ", numdatosP)
    count = 0
    # se crea una lista llamada usedIndex la cual tendra todos los indices ingresados a listas de prueba y entrenamiento
    usedIndex = []

    # listas de entrenamiento y de prueba
    training = []
    test = []

    # se toma un numero aleatorio y se compara si esta en la lista de indices ya tomados, en caso de no estar entonces \
    # se añade a la lista de entrenamiento
    while (count < numdatosT):
        randomNum = random.randrange(len(iris.data))  # ya que son 0 a 149 posiciones se toma el rango de 150 (el tamaño de la base de datos)
        if randomNum not in usedIndex:
            usedIndex.append(randomNum)
            training.append(list(iris.data[randomNum]))
            count += 1
    # Llenar test
    count = 0
    while (count < numdatosP):
        randomNum = random.randrange(len(iris.data))
        if randomNum not in usedIndex:
            usedIndex.append(randomNum)
            test.append(list(iris.data[randomNum]))
            count += 1
    return test, training


def crearListas_CV(iris):
    # se pide el numero de divisiones
    k = 9
    while (len(iris.data) % k != 0):
        k = input("Ingrese el numero de divisiones")
        k = int(k)
        if len(iris.data) % k != 0:
            print("Divisiones no validas ya que ",len(iris.data), "dividido", k, "da un flotante: ",len(iris.data) / k)
    print("numero total de datos: ", len(iris.data))
    print("numero de divisiones: ",k)
    print("numero de datos por division: ", len(iris.data) // k)
    setSize = len(iris.data) / k  # se calcula el tamaño de cada sublista
    training = []
    test = []
    ## Se llenan las sublistas y se ponen en training y test
    count = 0
    subCount = 0
    usedIndex = []
    subSet = []  # cada sublista, tendran tamaños de 15 si las divisiones son de 10
    while count < k:
        while subCount < setSize:
            randomNum = random.randrange(len(iris.data))
            if randomNum not in usedIndex:
                usedIndex.append(randomNum)
                subSet.append(list(iris.data[randomNum]))
                subCount += 1
        if count != k - 1:
            training.append(subSet)
            subSet = []
            count += 1
            subCount = 0
        else:
            test = subSet
            count += 1
    return test, training


def showData(test, training):
    choice = 0
    print("Desea mostrar los datos?")
    while choice != 1 and choice != 2:
        print("si  (1)")
        choice = input("no  (2)")
        choice = int(choice)
    if choice == 1:
        print("Datos en lista de training")
        for k in enumerate(training, start=1):
            print(k)

        print("Datos en lista de test")
        for k in enumerate(test, start=1):
            print(k)