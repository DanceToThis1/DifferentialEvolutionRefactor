import numpy as np
from paper.base import DifferentialEvolutionOptimizer
from functions import fun_sphere

class DE(DifferentialEvolutionOptimizer):
    """
    标准差分进化算法 (Standard Differential Evolution)
    继承自 DifferentialEvolutionOptimizer
    """
    def __init__(self, obj_func, bounds, mut=0.9, cr=0.1, pop_size=20, iterations=1000):
        super().__init__(obj_func, bounds, pop_size, iterations)
        self.mut = mut
        self.cr = cr

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            for j in range(self.pop_size):
                # 1. 变异 (Mutation)
                indexes = [index for index in range(self.pop_size) if index != j]
                a, b, c = self.population[np.random.choice(indexes, 3, replace=False)]
                mutant = a + self.mut * (b - c)
                mutant = self.check_bounds(mutant)
                
                # 2. 交叉 (Crossover)
                cross_points = np.random.rand(self.dimensions) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimensions)] = True
                trial = np.where(cross_points, mutant, self.population[j])
                
                # 3. 选择 (Selection)
                f = self.obj_func(trial)
                if f < self.fitness[j]:
                    self.fitness[j] = f
                    self.population[j] = trial
                    if f < self.fitness[self.best_index]:
                        self.best_index = j
                        self.best_vector = trial
            
            # 每次迭代返回当前最优解
            yield self.best_vector, self.fitness[self.best_index]

# 为了保持兼容性，保留之前的测试函数，但内部改为使用类
def de_rand_1_test(fun=None, bounds=None, mut=0.9, cr=0.1, pop_size=100, iterations=1000):
    if fun is None:
        fun = fun_sphere
    if bounds is None:
        bounds = [(-100, 100)] * 30
    
    # 实例化我们新写的类
    optimizer = DE(fun, bounds, mut, cr, pop_size, iterations)
    it = list(optimizer.run())
    print(it[-1][1])

if __name__ == '__main__':
    de_rand_1_test()