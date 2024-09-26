import numpy as np


def check_for_optimal_solution(F):
    return np.all(F >= 0)



def detect_leading_column(F):
    leading_column = np.argmin(F)
    return leading_column


eps = 1e-6

def calculate_min(A, b, leading_column):
    new_m = np.zeros((3,1))
    # print('b')
    # print(b)

    # print('A[:, leading_column]')
    # print(A[:, leading_column])

    for row in range(A.shape[0]):
            if abs(b[row, 0]) < eps:
                b[row, 0] = 0
            if abs(A[row, leading_column]) < eps:
                A[row, leading_column] = 0
            if A[row, leading_column] < -eps and  abs(b[row, 0]) < eps:
                new_m[row, 0] = None
            if abs(A[row, leading_column]) > eps:
                new_m[row, 0] = b[row, 0]/A[row, leading_column]
            else:
                new_m[row, 0] = None
    # print('new_m')
    # print(new_m)
    return new_m


def find_pivot_row(A, m, leading_column):
    minimum_positive_value = float('inf')
    minimum_indices = []

    for row in range(m.shape[0]):
        if m[row, 0] is None or m[row, 0] < 0:
            continue
        if m[row, 0] < minimum_positive_value:
            minimum_positive_value = m[row, 0]
            minimum_indices = [row]
        elif m[row, 0] == minimum_positive_value:
            minimum_indices.append(row)

    if len(minimum_indices) == 1:
        pirvot_row = minimum_indices[0]
        return pirvot_row
    
    # правило Креко
    # print('A')
    # print(A)
    # print('m')
    # print(m)
    # print(f'{minimum_indices=}')
    A2 = A[minimum_indices, :]
    for i in range(len(minimum_indices)):
        A2[i,:] = A2[i,:] / A2[i, leading_column]
    
    pirvot_row = minimum_indices[np.argmin(np.argmin(A2, axis=1))]
    return pirvot_row    