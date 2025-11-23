import numpy as np

class DifferentialEvolutionOptimizer:
    """
    差分进化算法基类 (Base Class)
    封装了初始化、边界检查、适应度评估以及通用的变异/交叉策略。
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
        
        self.population = None
        self.fitness = None
        self.best_index = -1
        self.best_vector = None

    def initialize_population(self):
        pop = np.random.rand(self.pop_size, self.dimensions)
        self.population = self.min_bound + pop * self.diff
        self.fitness = np.asarray([self.obj_func(ind) for ind in self.population])
        self.best_index = np.argmin(self.fitness)
        self.best_vector = self.population[self.best_index]

    def check_bounds(self, mutant):
        return np.clip(mutant, self.min_bound, self.max_bound)

    # --- 通用变异与交叉策略 (Strategies) ---

    def strategy_rand_1_bin(self, target_idx, mut, cr):
        """策略 0: DE/rand/1/bin"""
        indexes = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indexes, 3, replace=False)]
        
        mutant = a + mut * (b - c)
        mutant = self.check_bounds(mutant)
        
        cross_points = np.random.rand(self.dimensions) < cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
        
        return np.where(cross_points, mutant, self.population[target_idx])

    def strategy_rand_to_best_2_bin(self, target_idx, mut, cr):
        """策略 1: DE/rand-to-best/2/bin"""
        indexes = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c, d = self.population[np.random.choice(indexes, 4, replace=False)]
        
        # 变异向量基于当前个体、最优个体以及两个差分向量
        target = self.population[target_idx]
        mutant = target + mut * (self.best_vector - target) + mut * (a - b) + mut * (c - d)
        mutant = self.check_bounds(mutant)
        
        cross_points = np.random.rand(self.dimensions) < cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
            
        return np.where(cross_points, mutant, target)

    def strategy_rand_2_bin(self, target_idx, mut, cr):
        """策略 2: DE/rand/2/bin"""
        indexes = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c, d, e = self.population[np.random.choice(indexes, 5, replace=False)]
        
        mutant = a + mut * (b - c) + mut * (d - e)
        mutant = self.check_bounds(mutant)
        
        cross_points = np.random.rand(self.dimensions) < cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
            
        return np.where(cross_points, mutant, self.population[target_idx])

    def strategy_current_to_rand_1(self, target_idx, mut):
        """策略 3: DE/current-to-rand/1 (无显式交叉)"""
        indexes = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indexes, 3, replace=False)]
        target = self.population[target_idx]
        
        k = np.random.rand() # 这里的 K 是随机权重
        # 这里的实现不进行 bin 交叉，而是直接生成 trial
        trial = target + k * (a - target) + mut * (b - c)
        return self.check_bounds(trial)

    def run(self):
        raise NotImplementedError("Subclasses should implement this!")