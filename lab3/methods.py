# Метод дихтомии
def dichotomy(target_func, a, b, eps):
    n_calls = 0
    while abs(b - a) > eps:
        x = (a + b) / 2
        delta = eps * 1e-3  # малая величина дельта для 
        
        fx_left = target_func(x - delta); n_calls += 1
        fx_right = target_func(x + delta); n_calls += 1
        
        if fx_left > fx_right:
            a = x
        else:
            b = x
    return a, b, n_calls


# Метод золотого сечения
def golden_ratio(target_func, a, b, eps):
    n_calls = 0
    phi = (1 + 5 ** 0.5) / 2  # золотое сечение
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi

    fx1 = target_func(x1); n_calls += 1
    fx2 = target_func(x2); n_calls += 1

    while abs(b - a) > eps:
        if fx1 < fx2:
            b = x2
            x2 = x1
            x1 = b - (b - a) / phi
            fx2 = fx1

            fx1 = target_func(x1); n_calls += 1

        else:
            a = x1
            x1 = x2
            x2 = a + (b - a) / phi
            fx1 = fx2

            fx2 = target_func(x2); n_calls += 1
            
    return a, b, n_calls