from sklearn.datasets import load_iris
iris = load_iris()
targets = iris.target # llena targets con los tipos (clases) de las flores
flowers = []
for k in iris.data: # llena flowers con los elementos de iris.data
    temp = list(k)
    flowers.append(temp)


def dataForNormalization():
    mini = []
    maxi = []

    for i in range(len(flowers[0])):
        mini.append(min(x[i] for x in flowers))
        maxi.append(max(x[i] for x in flowers))
    return mini, maxi

min, max = dataForNormalization()


def normalization(x, y, min , max):

    x = (x - min) / (max - min)
    y = (y - min) / (max - min)

    return x, y


def getDistance(x, y):
    sum = 0
    for i, j, mini, maxi in zip(x, y, min, max):
        i, j = normalization(i, j, mini, maxi)  #se normalizan los valores antes de sacar la distancia
        if i and j is not None:
            operation = i - j
            if operation < 0:
                operation *= -1
            sum += operation
    return sum



