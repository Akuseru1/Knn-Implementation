#normalizar

def getDistance(x, y):
    sum = 0
    for i, j in zip(x, y):
        if i and j is not None:
            operation = i - j
            if operation < 0:
                operation *= -1
            sum += operation
    return sum



