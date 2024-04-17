import numpy as np
import matplotlib.pyplot as plt

from methods import *

def plot_target(funcs, a, b):
    x = np.linspace(a, b, 200)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # список цветов для графиков

    plt.figure(figsize=(8, 6))
    legends = []  # список для легенды

    for i, f in enumerate(funcs):
        y = np.vectorize(f)(x)
        plt.plot(x, y, colors[i % len(colors)], linewidth=2, label=f'f{i}(x)')
        legends.append(f)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('График функций цели')
    plt.grid(True)
    plt.xlim(a, b)
    plt.legend(legends)
    plt.show()

def plot_compare(target_func, a, b, eps_list):
    dichotomy_steps = []
    golden_ratio_steps = []

    for eps in eps_list:
        a_d, b_d, n_d = dichotomy(target_func, a, b, eps)
        a_g, b_g, n_g = golden_ratio(target_func, a, b, eps)
        dichotomy_steps.append(n_d)
        golden_ratio_steps.append(n_g)

    plt.figure(figsize=(8, 6))
    plt.plot(dichotomy_steps, eps_list, 'bo-', label='Метод дихотомии')
    plt.plot(golden_ratio_steps, eps_list, 'rv--', label='Метод золотого сечения')
    plt.yscale('log')
    plt.xlabel('Число вызовов функции')
    plt.ylabel('eps')
    # plt.title('Сравнение числа вызовов функции')
    plt.legend()
    plt.grid(True)
    plt.show()