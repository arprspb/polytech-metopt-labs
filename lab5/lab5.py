import numpy as np

# Функция цели
def objective_function(x, y, z):
    return -(x + y + z)

# Ограничения
def constraint1(x, y, z):
    return max(0, 10*x + y + 8*z - 20)
def constraint2(x, y):
    return max(0, y - 2*x)
def constraint3(x, y, z):
    return abs(2*z - x - y)

# Вспомогательная функция с штрафными членами
def penalty_function(x, y, z, r):
    return objective_function(x, y, z) + r * (constraint1(x, y, z) + constraint2(x, y) + constraint3(x, y, z))

def gradient_phi1(x, y, z):
    grad = np.zeros(3)
    if constraint1(x, y, z) > 0:
        grad[0] = 10
        grad[1] = 1
        grad[2] = 8
    return grad

def gradient_phi2(x, y, z):
    grad = np.zeros(3)
    if constraint2(x, y) > 0:
        grad[0] = -2
        grad[1] = 1
    return grad

def gradient_phi3(x, y, z):
    grad = np.zeros(3)
    if constraint3(x, y, z) > 0:
        grad[0] = -1
        grad[1] = -1
        grad[2] = 2
    return grad


def penalty_function_gradient(x, y, z, r):
    grad_obj = np.array([-1, -1, -1])  # Градиент целевой функции
    
    grad_phi1 = gradient_phi1(x, y, z)
    grad_phi2 = gradient_phi2(x, y, z)
    grad_phi3 = gradient_phi3(x, y, z)
    
    grad = grad_obj + r * (grad_phi1 + grad_phi2 + grad_phi3)
    return grad

def gradient_descent(x0, y0, z0, alpha, beta, gradient_function, max_iter=100, epsilon=1e-6):
    """
    Функция градиентного спуска для минимизации задачи с ограничениями.

    Параметры:
    x0, y0, z0 - начальные значения переменных
    alpha - шаг градиентного метода
    gradient_function - функция градиента целевой функции
    max_iter - максимальное число итераций
    epsilon - точность метода

    Возвращает:
    x_opt, y_opt, z_opt - оптимальные значения переменных
    iterations - количество итераций
    """

    x_opt = x0
    y_opt = y0
    z_opt = z0
    iterations = 0
    r = 1

    while iterations < max_iter:
        grad_x, grad_y, grad_z = gradient_function(x_opt, y_opt, z_opt, r)

        x_opt -= alpha * grad_x
        y_opt -= alpha * grad_y
        z_opt -= alpha * grad_z

        if np.linalg.norm([grad_x, grad_y, grad_z]) < epsilon:
            break

        iterations += 1
        r *= beta

    return x_opt, y_opt, z_opt, iterations



def main():
    # Начальные значения переменных
    x_start, y_start, z_start = 0, 0, 0
    
    # Параметры метода
    alpha = 0.01
    max_iterations = 1000
    epsilon = 1e-6
    
    # Параметр штрафа
    beta = 2
    
    # Градиентный метод наискорейшего спуска
    result = gradient_descent(
        x_start,
        y_start,
        z_start,
        alpha,
        beta,
        penalty_function_gradient,
        max_iter=max_iterations,
        epsilon=epsilon)
    
    # Результаты оптимизации
    x_opt, y_opt, z_opt, iterations = result
    result_value = objective_function(x_opt, y_opt, z_opt)
    
    print("Результаты оптимизации:")
    print(f"Значение x: {x_opt}")
    print(f"Значение y: {y_opt}")
    print(f"Значение z: {z_opt}")
    print(f"Значение целевой функции: {result_value}")
    print(f"Количество итераций: {iterations}")


if __name__ == "__main__":
    main()