from Neural_Network.Training import backPropagation, getValues

def show():
    print(getValues())
    for i in range(100):
        backPropagation()
    print(getValues())