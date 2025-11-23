import numpy as np
import random
from scipy.stats import cauchy
import matplotlib.pyplot as plt
from paper.base import DifferentialEvolutionOptimizer

class JADE(DifferentialEvolutionOptimizer):
    """
    JADE: Adaptive Differential Evolution with Optional External Archive
    """
    def __init__(self, obj_func, bounds, pop_size=100, iterations=1000, c=0.1, p=0.05):
        super().__init__(obj_func, bounds, pop_size, iterations)
        self.c = c  # 参数适应的学习率
        self.p = p  # 选取前 p% 的最优个体作为变异基准
        self.mean_cr = 0.5
        self.mean_mut = 0.5
        self.archive = [] # 外部存档

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            # 1. 种群排序：为了选出前 p% 的个体 (x_best_p)
            # 原代码是用 obj_func 重新计算一遍来排序
            # 直接利用已有的 self.fitness 进行排序，速度会快很多。
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]
            
            # 更新当前全局最优（排序后第一个就是最优）
            self.best_index = 0
            self.best_vector = self.population[0]
            
            success_mut = []
            success_cr = []
            population_new = np.copy(self.population)
            fitness_new = np.copy(self.fitness)
            
            # 2. 生成子代
            for j in range(self.pop_size):
                # 选择 p-best (前 p*pop_size 个体中随机选一个)
                top_p_count = max(1, int(self.p * self.pop_size))
                p_best_idx = random.randint(0, top_p_count - 1)
                x_best_p = self.population[p_best_idx]
                
                # 生成自适应参数 CR 和 F (mut)
                cr = np.clip(random.gauss(self.mean_cr, 0.1), 0, 1)
                
                # F 服从柯西分布，如果在 (0, 1] 区间外则重生成或截断
                while True:
                    mut = cauchy.rvs(loc=self.mean_mut, scale=0.1)
                    if mut >= 1:
                        mut = 1
                        break
                    if mut > 0:
                        break
                        
                # 调用父类的通用策略
                trial = self.strategy_current_to_pbest_1_bin_with_archive(
                    j, x_best_p, mut, cr, self.archive
                )
                
                # 选择
                f_trial = self.obj_func(trial)
                
                if f_trial < self.fitness[j]:
                    population_new[j] = trial
                    fitness_new[j] = f_trial
                    
                    # 成功个体的参数记录下来用于更新
                    success_cr.append(cr)
                    success_mut.append(mut)
                    
                    # 将被淘汰的父代加入存档
                    self.archive.append(self.population[j].copy())
                else:
                    population_new[j] = self.population[j]
                    fitness_new[j] = self.fitness[j]
            
            # 3. 维护存档大小 (不能超过 pop_size)
            while len(self.archive) > self.pop_size:
                # 随机移除
                remove_idx = random.randint(0, len(self.archive) - 1)
                self.archive.pop(remove_idx)
            
            # 4. 更新种群
            self.population = population_new
            self.fitness = fitness_new
            
            # 5. 更新自适应参数 mean_cr 和 mean_mut
            if len(success_cr) > 0:
                self.mean_cr = (1 - self.c) * self.mean_cr + self.c * np.mean(success_cr)
                # Lehmer mean for F
                sum_f2 = sum(f**2 for f in success_mut)
                sum_f = sum(success_mut)
                if sum_f > 0: # 避免除零
                    lehmer_mean = sum_f2 / sum_f
                    self.mean_mut = (1 - self.c) * self.mean_mut + self.c * lehmer_mean
            
            yield self.best_vector, self.fitness[0]

# 保持接口兼容
def jade_test_50(fun, bounds, iterations):
    result = []
    for num in range(50):
        optimizer = JADE(fun, bounds, pop_size=100, iterations=iterations)
        it = list(optimizer.run())
        result.append(it[-1][-1])
        print(num, result[-1])
    print(f"Mean: {np.mean(result)}")

if __name__ == '__main__':
    from functions import fun_sphere
    temp_optimizer = JADE(fun_sphere, [(-100, 100)] * 30, pop_size=100, iterations=1000)
    it = list(temp_optimizer.run())
    print(it[-1][1])