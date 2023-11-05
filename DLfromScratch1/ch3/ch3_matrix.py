import numpy as np

def dot_product(A, B):
    return np.dot(A, B)

def dot_product_without_np(A, B):
    n1, m1 = A.shape
    n2, m2 = B.shape
    C = np.zeros([n1,m2])

    if m1 == n2:
        for i in range(n1):
            for j in range(m1):
                for k in range(m2):
                    C[i][k] += A[i][j]*B[j][k]
        return C            
    else:
        raise ValueError

A = np.array([[1,2, 3], 
              [4, 5, 6], 
              ])
B = np.array([[1, 2, 3], 
              [3, 4, 5], 
              [5, 6,7]]
              )


def main():
    # Dot product in NeuralNetwork
    X = np.array([1,2])
    W = np.array([[1,3,5], [2,4,6]])
    Y = np.dot(X, W)
    print(Y)


if __name__ == '__main__':
    main()