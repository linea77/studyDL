import pickle
import numpy as np

from ch3_dataset_mnist import load_mnist
from studyDL.DLfromScratch1.activation_functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network=pickle.load(f) # weights, bias
    return network

def predict(network:dict, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3)+b3
    z3 = softmax(a3)
    return z3


def get_accuracy(x, t, network):

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x)
        p = np.argmax(y)
        if p == t[1]:
            accuracy_cnt += 1
    return (accuracy_cnt / len(x)) 


def main():
    x, t = get_data()
    network = init_network()

    accuracy = get_accuracy
    print(accuracy)


if __name__ == '__main__':
    main()