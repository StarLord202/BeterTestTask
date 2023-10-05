import math
import numpy as np


def tree_calculator(p):

    if p<0 or p>1:
        raise ValueError

    q = 1-p

    prob = (p*p)/(1-2*p*q)

    return prob

def matrix_calculator(p):

    if p<0 or p>1:
        raise ValueError

    q = 1-p

    matrix = np.array([[0, p, q, 0, 0, 0],
                         [0, 0, 0, p, q, 0],
                         [0, 0, 0, 0, p, q],
                         [0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]])


    eps = 3
    n = int((eps + 2*math.log(p, 10)) / (-1 * math.log(2*p*q, 10))) + 1


    matrix_3 = np.linalg.matrix_power(matrix, 3)
    matrix_accumulator = np.linalg.matrix_power(matrix, 2)

    prob_accumulator = matrix_accumulator[0, 3]


    for _ in range(n):
        matrix_accumulator = matrix_accumulator @ matrix_3
        prob_accumulator += matrix_accumulator[0, 3]

    return prob_accumulator


def main():
    p = 0.5

    res1 = tree_calculator(p)
    res2 = matrix_calculator(p)

    print(f"prob calculated with tree: {res1}")
    print(f"prob calculated with markov: {res2}")
    print(f"difference between them: {abs(res1 - res2)}")

if __name__ == '__main__':
    main()







