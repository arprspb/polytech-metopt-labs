import numpy as np

def simplex_method(A, b, c, basis_indices, var_names_dict, minimize=True):
    var_names_reverse = {v: k for k, v in var_names_dict.items()}
    m, n = A.shape

    # Инвертируем цель, если минимизация

    # Создаем начальную таблицу
    tableau = np.zeros((m + 1, n + 1))
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b

    # Считаем дельты
    ce = c[basis_indices]
    deltas = np.dot(ce, A) - c
    delta_b = np.dot(ce, b)
    tableau[-1, :-1] = deltas
    tableau[-1, -1] = delta_b

    iteration = -1
    while True:
        iteration += 1
        current_basis = ", ".join([var_names_reverse[i] for i in basis_indices])
        print(f"Итерация {iteration}, базисные переменные: {current_basis}")

        print("\n".join(["\t".join(map("{:0.2f}".format, row)) for row in tableau]))
        
        # Шаг 1: Поиск разрешающего столбца
        if minimize:
            pivot_col = np.argmax(np.maximum(tableau[-1, :-1], 0))
            if tableau[-1, pivot_col] <= 0:
                break  # Оптимальное решение найдено
        else:
            pivot_col = np.argmin(np.minimum(tableau[-1, :-1], 0))
            if tableau[-1, pivot_col] >= 0:
                break  # Оптимальное решение найдено
        

        # Шаг 2: Поиск разрешающей строки
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        if minimize:
            ratios[ratios < 0] = np.inf
            pivot_row = np.argmin(ratios)
        else:
            ratios[ratios > 0] = -1 * np.inf
            pivot_row = np.argmax(ratios)

        basis_indices[pivot_row] = pivot_col


        # Шаг 3: Обновление таблицы
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[pivot_row, :] * tableau[i, pivot_col]

    # Результаты
    solution = np.zeros(n)
    for row, basis in enumerate(basis_indices):
        solution[basis] = tableau[row, -1]

    # Возвращаем оптимальное значение целевой функции с учетом знака
    optimal_value = tableau[-1, -1]
    return solution, optimal_value
