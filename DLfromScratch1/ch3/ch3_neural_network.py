import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Network:
    def __init__(self):
        # layer1 : 2X3
        self.W1 = np.array([[0.1, 0.3, 0.5],
                            [0.2, 0.4, 0.6]])
        self.b1 = np.array([0.1, 0.2, 0.3])
        
        # layer2: 3X2
        self.W2 = np.array([[0.1, 0.4],
                            [0.2, 0.5],
                            [0.3, 0.6]])
        self.b2 = np.array([0.1, 0.2])

        #layer 3: 2X2
        self.W3 = np.array([[0.1, 0.3],
                            [0.2, 0.4]])
        self.b3 = np.array([0.1, 0.2])

    def forward(self, x):
        a1 = np.dot(x, self.W1) + self.b1
        z1 = sigmoid(a1)
        
        a2 = np.dot(z1, self.W2) + self.b2
        z2 = sigmoid(a2)

        a3 = np.dot(z2, self.W3) + self.b3
        y = a3
        return y
    

def main():
    network = Network()
    x = np.array([ 1.0, 0.5])
    y = network.forward(x)
    print(y)


if __name__ == '__main__':
    main()

        

