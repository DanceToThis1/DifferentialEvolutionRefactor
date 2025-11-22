import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
在基本算法基础上改进参数选择策略。
"""


def jde(fobj, bounds, mut=0.9, cr=0.1, pop_size=100, iterations=1000):
    dimensions = len(bounds)
    pop = np.random.rand(pop_size, dimensions)
    min_bound, max_bound = np.asarray(bounds).T
    diff = np.fabs(min_bound - max_bound)
    population = min_bound + pop * diff
    fitness = np.asarray([fobj(ind) for ind in population])
    best_index = np.argmin(fitness)
    best = population[best_index]
    for i in range(iterations):
        for j in range(pop_size):
            indexes = [index for index in range(pop_size) if index != j]
            a, b, c = population[np.random.choice(indexes, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), min_bound, max_bound)
            cross_points = np.random.rand(dimensions) < cr
            randj = np.random.rand(4)
            if randj[0] < 0.1:
                mut = 0.1 + randj[1] * 0.9
            if randj[2] < 0.1:
                cr = randj[3]
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, population[j])
            f = fobj(trial)
            if f < fitness[j]:
                fitness[j] = f
                population[j] = trial
                if f < fitness[best_index]:
                    best_index = j
                    best = trial
        yield best, fitness[best_index]


def jde_test(fun, bounds, mut=0.9, cr=0.1, iterations=3000, log=0):
    it = list(jde(fun, bounds, mut=mut, cr=cr, pop_size=100, iterations=iterations))
    print(it[-1])
    x, f = zip(*it)
    plt.plot(f, label='jde')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    plt.show()


def jde_test_50(fun, bounds, iterations):
    result = []
    for num in range(50):
        it = list(jde(fun, bounds, pop_size=100, iterations=iterations))
        result.append(it[-1][-1])
        print(num, result[-1])
    # 由于删除了 path1，下面两行无法直接运行，下一步重构中修复路径问题。
    # data = pd.DataFrame([['JDE', fun.__name__, its, i] for i in result])
    # data.to_csv(path1 + '/all_algorithm_test_data/data.csv', mode='a', header=False)
    mean_result = np.mean(result)
    std_result = np.std(result)
    # data_mean = pd.DataFrame([['JDE', fun.__name__, its, mean_result, std_result]])
    # data_mean.to_csv(path1 + '/all_algorithm_test_data/data.csv', mode='a', index=False, header=False)
