from Neural_Network.Training import updating, getValues

def show():
    print(getValues())
    for i in range(10):
        updating()
    print(getValues())