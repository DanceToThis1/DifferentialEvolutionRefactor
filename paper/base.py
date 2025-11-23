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

    def strategy_current_to_pbest_1_bin_with_archive(self, target_idx, x_pbest, mut, cr, archive=None):
        """
        策略: DE/current-to-pbest/1/bin (带存档支持)
        公式: v = x_current + F * (x_pbest - x_current) + F * (x_r1 - x_r2)
        注意: x_r2 可以从 (当前种群 + 存档) 的并集中选择
        """
        target = self.population[target_idx]
        pop_len = self.pop_size
        
        # 1. 选择 r1 (不能是自己)
        r1 = np.random.randint(0, pop_len)
        while r1 == target_idx:
            r1 = np.random.randint(0, pop_len)
        x_r1 = self.population[r1]
        
        # 2. 选择 r2 (从 种群 U 存档 中选择，且不能是自己或 r1)
        if archive is not None and len(archive) > 0:
            archive_len = len(archive)
            total_len = pop_len + archive_len
            
            r2 = np.random.randint(0, total_len)
            # 如果 r2 对应的个体是自己或 r1，重选
            # (注意：这里简化了判断，严格来说需要判断向量内容是否相等，但索引判断效率更高)
            while r2 == target_idx or r2 == r1:
                r2 = np.random.randint(0, total_len)
            
            if r2 < pop_len:
                x_r2 = self.population[r2]
            else:
                x_r2 = archive[r2 - pop_len]
        else:
            # 没有存档时的逻辑
            r2 = np.random.randint(0, pop_len)
            while r2 == target_idx or r2 == r1:
                r2 = np.random.randint(0, pop_len)
            x_r2 = self.population[r2]
            
        # 3. 变异计算
        mutant = target + mut * (x_pbest - target) + mut * (x_r1 - x_r2)
        mutant = self.check_bounds(mutant)
        
        # 4. 交叉
        cross_points = np.random.rand(self.dimensions) < cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
            
        return np.where(cross_points, mutant, target)

    def strategy_best_2_bin(self, target_idx, mut, cr):
        """策略: DE/best/2/bin"""
        indexes = [i for i in range(self.pop_size) if i != target_idx]
        a, b, c, d = self.population[np.random.choice(indexes, 4, replace=False)]
        
        # 基向量是全局最优 best_vector
        mutant = self.best_vector + mut * (a - b) + mut * (c - d)
        mutant = self.check_bounds(mutant)
        
        cross_points = np.random.rand(self.dimensions) < cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dimensions)] = True
            
        return np.where(cross_points, mutant, self.population[target_idx])
    
    def run(self):
        raise NotImplementedError("Subclasses should implement this!")