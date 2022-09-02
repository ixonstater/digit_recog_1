from network import *
from random import random
from math import floor
import numpy as np
from multiprocessing import Process

def readTrainingInputs():
    inFile = open('./test_data/inputs.txt', 'r')
    inputs = []
    singleInputs = []
    targets = []
    for line in inFile:
        if(line == '\n'):
            inputs.append(singleInputs)
            singleInputs = []
            continue
        elif (len(line) == 2):
            tar = [0.1] * 5
            tar[int(line[0])] = 0.9
            targets.append(tar)
        else:
            for char in range(0, 8):
                singleInputs.append(int(line[char]))
    return inputs, targets

def ppInputs(arr, file=None):
    for i in range(0, 9):
        for j in range(0, 8):
            if(arr[i * 8 + j] == 0):
                print('   ', end = '', file=file)
            else:
                print('*  ', end = '', file=file)
        print('\n', file=file)

def randomIndex(ceil):
    return floor(random() * ceil)

def interpretOutputs(outs):
    return(np.argmax(outs))

def demo(inputs, targets, net):
    samples = 80000
    for i in range(0, samples):
        if (i % 1000 == 0):
            remaining = int((samples - i) / 1000)
            done = int(i / 1000)
            if i > 0:
                print ("\033[A\033[A")
            print('<' + '*' * done + ' ' * remaining + '>')
        indx = randomIndex(339)
        net.backprop(inputs[indx], targets[indx])
    file = open("outputs.txt", "w+")
    for i in range(0, 339):
        outs = net.feedForward(inputs[i])[1]
        digit = interpretOutputs(outs)
        print("I thought this was a: ", digit, file=file)
        print("This is really a: ", interpretOutputs(targets[i]), file=file)
        ppInputs(inputs[i], file=file)
    file.close()
    print("Predictions written to outputs.txt file.")

def trainAndTest():
    myNetwork = Network([72, 10, 5], 0.05, 0.3)
    inputs, targets = readTrainingInputs()
    for i in range(0, 100000):
        indx = randomIndex(339)
        myNetwork.backprop(inputs[indx], targets[indx])
    right = 0
    wrong = 0
    for i in range(0, 339):
        outs = myNetwork.feedForward(inputs[i])[1]
        guessDigit = interpretOutputs(outs)
        realDigit = interpretOutputs(targets[i])
        if(realDigit == guessDigit):
            right += 1
        else:
            wrong += 1
    return myNetwork, wrong

def bestOfTen():
    bestWeights = []
    bestBiases = []
    leastWrong = 339
    for i in range(0, 10):
        myNetwork, wrong = trainAndTest()
        if(wrong < leastWrong):
            bestWeights = myNetwork.weights
            bestBiases = myNetwork.biases
            leastWrong = wrong
        print(wrong)
    np.save('./test_data/wb_ryans_data/weights.npy', bestWeights)
    np.save('./test_data/wb_ryans_data/biases.npy', bestBiases)

