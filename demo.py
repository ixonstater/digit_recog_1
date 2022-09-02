import digit_recog
import network

if __name__ == "__main__":
    inputs, targets = digit_recog.readTrainingInputs()
    net = network.Network([72, 20, 5])
    digit_recog.demo(inputs, targets, net)