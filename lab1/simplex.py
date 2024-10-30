import numpy as np

def simplex_method(A, b, c, basis_indices):
    # Обновляем количество переменных
    m, n = A.shape

    # Создаем начальную таблицу
    tableau = np.zeros((m + 1, n + 1))
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b

    # Считаем дельты
    ce = c[basis_indices]  # Коэффициенты целевой функции для базисных переменных
    deltas = np.dot(ce, A) - c
    delta_b = np.dot(ce, b)
    tableau[-1, :-1] = deltas
    tableau[-1, -1] = delta_b  # Последний элемент строки дельт — свободный член

    iteration = -1
    while True:
        iteration += 1
        current_basis = ", ".join([var_names_reverse[i] for i in basis_indices])
        print(f"Итерация {iteration}, базисные переменные: {current_basis}")
        
        # Шаг 1: Поиск разрешающего столбца
        pivot_col = np.argmax(np.maximum(tableau[-1, :-1], 0))
        if tableau[-1, pivot_col] <= 0:
            break  # Оптимальное решение найдено

        # Шаг 2: Поиск разрешающей строки
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios < 0] = np.inf  # Игнорируем отрицательные
        pivot_row = np.argmin(ratios)

        basis_indices[pivot_row] = pivot_col

        # Шаг 3: Обновление таблицы
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element  # Нормализация разрешающей строки

        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[pivot_row, :] * tableau[i, pivot_col]


    

    # Результаты
    solution = np.zeros(n)
    for row, basis in enumerate(basis_indices):
        solution[basis] = tableau[row, -1]
    print("\n".join(["\t".join(map("{:0.2f}".format, row)) for row in tableau]))
    
    return solution, tableau[-1, -1]



A = np.array([[-1, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [-1, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0],
              [-1, 3, 2, 0, 0, 1, 0, 0, 0, 0, 0],
              [-1, 2, 3, 0, 0, 0, 1, 0, 0, 0, 0],
              [-1, 1, 4, 0, 0, 0, 0, 1, 0, 0, 0],
              [-1, 0, 5, 0, 0, 0, 0, 0, 1, 0, 0],
              [ 0, 3, 2, 0, 0, 0, 0, 0, 0, -1, 1]
            ])


b = np.array([400, 
              400,
              400,
              400,
              400,
              400,
              800
              ])

c = np.array([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e3])  # Большой коэффициент для u

var_names_dict = {
    "x1": 0,
    "x2": 1,
    "x3": 2,
    "x4": 3,
    "x5": 4,
    "x6": 5,
    "x7": 6,
    "x8": 7,
    "x9": 8,
    "x10": 9,
    "u1": 10
}

var_names_reverse = {v: k for k, v in var_names_dict.items()}

basis_indices = list(map(var_names_dict.get, ["x4", "x5", "x6", "x7", "x8", "x9", "u1"]))

solution, optimal_value = simplex_method(A, b, c, basis_indices)

print("Решение:", solution)
print("Оптимальное значение целевой функции:", optimal_value)
