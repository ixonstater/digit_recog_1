import numpy as np
from math import exp

class Network:
    def __init__(self, networkSpecs, learningRate = 1, momentumRate = 0):
        self.l1N, self.l2N, self.l3N = networkSpecs
        weights = np.array([np.random.rand(self.l2N, self.l1N), np.random.rand(self.l3N, self.l2N)], dtype=object)
        self.weights = (weights - 0.5) * 0.1
        self.biases = np.array([np.random.rand(self.l2N), np.random.rand(self.l3N)], dtype=object)
        self.learningRate = learningRate
        self.momentumRate = momentumRate
        self.prevWeightUpdates = np.array([np.zeros((self.l2N, self.l1N)), np.zeros((self.l3N, self.l2N))], dtype=object)
        self.prevBiasUpdates = np.array([np.zeros([self.l2N]), np.zeros([self.l3N])], dtype=object)

    def feedForward(self, inputs):
        hiddenOutputs = np.array(list(map(self.sigmoidFunc, [(self.weights[0][i], inputs, self.biases[0][i]) for i in range(0, len(self.biases[0]))])))
        finalOutputs = np.array(list(map(self.sigmoidFunc, [(self.weights[1][i], hiddenOutputs, self.biases[1][i]) for i in range(0, len(self.biases[1]))])))
        return hiddenOutputs, finalOutputs

    def backprop(self, inputs, targets):
        hiddenOutputs, finalOutputs = self.feedForward(inputs)
        outputErrorTerms = finalOutputs * (1 - finalOutputs) * (targets - finalOutputs)
        hiddenErrorTerms = hiddenOutputs * (1 - hiddenOutputs) * np.array(list(map(self.dotProduct, [(weights, outputErrorTerms) for weights in np.transpose(self.weights[1])])))
        self.updateWeights(outputErrorTerms, hiddenErrorTerms, inputs, hiddenOutputs)
        return self.calcTotErr(targets, finalOutputs), targets, finalOutputs

    def updateWeights(self, outErrTer, hidErrTer, ins, hidOuts):
        deltaOut = np.array(np.meshgrid(outErrTer, hidOuts)).T.reshape(-1, 2)
        deltaOut = np.array(list(map(lambda arg: arg[0] * arg[1], deltaOut))).reshape(self.l3N, self.l2N)
        deltaHid = np.array(np.meshgrid(hidErrTer, ins)).T.reshape(-1, 2)
        deltaHid = np.array(list(map(lambda arg: arg[0] * arg[1], deltaHid))).reshape(self.l2N, self.l1N)
        weightUpdates = np.array([deltaHid, deltaOut], dtype=object) * self.learningRate + self.prevWeightUpdates * self.momentumRate
        self.weights += weightUpdates
        self.prevWeightUpdates = weightUpdates
        biasUpdates = np.array([hidErrTer, outErrTer], dtype=object) * self.learningRate + self.momentumRate * self.prevBiasUpdates
        self.biases += biasUpdates
        self.prevBiasUpdates = biasUpdates

    def dotProduct(self, arg):
        return np.dot(arg[0], arg[1])

    def sigmoidFunc(self, arg):
        negY = -1 * (np.dot(arg[0], arg[1]) + arg[2])
        try:
            return 1 / (1 + exp(negY))
        except OverflowError:
            if(negY > 0):
                return 0
            else:
                return 1

    def calcTotErr(self, targets, outputs):
        difference = targets - outputs
        return(0.5 * np.dot(difference, difference))
