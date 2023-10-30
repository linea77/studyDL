import numpy as np


# def AND(x1, x2):
#     w1, w2, theta = 1.0, 1.0, 1.0
#     temp = x1*w1+x2*w2
#     if temp > theta:
#         return 1
#     else :
#         return 0

class PerceptronGate():
    def __init__(self, x1, x2):
        self.x = np.array([x1, x2])

    def forward(self, w, b):
        temp = np.sum(self.x*w)+b
        if temp > 0:
            return 1
        else:
            return 0
        
    def AND(self):
        w = np.array([1.0, 1.0])
        b = -1.0
        return self.forward(w, b)
    
    def NAND(self):
        w = np.array([-1.0, -1.0])
        b = 1.0
        return self.forward(w, b)
    
    def OR(self):
        w = np.array([1.0, 1.0])
        b = 0
        return self.forward(w, b)
    
    def XOR(self):
        x1 = self.AND()
        x2 = self.NAND()
        self.x = np.array([x1, x2])
        if (self.OR() == 1):
            return 0
        else:
            return 1






if __name__ == '__main__':
    
    for x1 in range(2):
        for x2 in range(2):
            gate = PerceptronGate(x1, x2)
            print(x1, x2, gate.XOR())