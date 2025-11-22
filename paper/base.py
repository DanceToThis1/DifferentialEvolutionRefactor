import numpy as np

class DifferentialEvolutionOptimizer:
    """
    差分进化算法基类
    封装了通用的初始化、边界检查和适应度评估逻辑。
    """
    def __init__(self, obj_func, bounds, pop_size=20, iterations=1000):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.min_bound = self.bounds[:, 0]
        self.max_bound = self.bounds[:, 1]
        self.diff = np.fabs(self.min_bound - self.max_bound)
        self.dimensions = len(bounds)
        self.pop_size = pop_size
        self.iterations = iterations
        
        # 初始化状态变量
        self.population = None
        self.fitness = None
        self.best_index = -1
        self.best_vector = None

    def initialize_population(self):
        """初始化种群并计算初始适应度"""
        pop = np.random.rand(self.pop_size, self.dimensions)
        self.population = self.min_bound + pop * self.diff
        self.fitness = np.asarray([self.obj_func(ind) for ind in self.population])
        self.best_index = np.argmin(self.fitness)
        self.best_vector = self.population[self.best_index]

    def check_bounds(self, mutant):
        """边界处理：默认使用截断法 (clip)"""
        return np.clip(mutant, self.min_bound, self.max_bound)

    def run(self):
        """主循环，由子类具体实现"""
        raise NotImplementedError("Subclasses should implement this!")