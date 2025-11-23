import numpy as np
import random
import pandas as pd
from scipy.stats import cauchy
import matplotlib.pyplot as plt
from paper.base import DifferentialEvolutionOptimizer

class SHADE(DifferentialEvolutionOptimizer):
    """
    SHADE: Success-History Based Linear Population Size Reduction Differential Evolution
    (这里只实现了参数自适应部分，未包含 L-SHADE 的种群线性减小)
    """
    def __init__(self, obj_func, bounds, pop_size=100, iterations=1000, memory_size=100):
        super().__init__(obj_func, bounds, pop_size, iterations)
        self.memory_size = memory_size
        # 历史记忆库初始化为 0.5
        self.memory_cr = np.full(memory_size, 0.5)
        self.memory_f = np.full(memory_size, 0.5)
        self.k_idx = 0 # 当前更新记忆库的位置指针
        self.archive = []

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            # 排序
            sorted_indices = np.argsort(self.fitness)
            self.population = self.population[sorted_indices]
            self.fitness = self.fitness[sorted_indices]
            self.best_vector = self.population[0]
            
            success_cr = []
            success_f = []
            diff_fitness = [] # 记录适应度提升量，作为权重
            
            population_new = np.copy(self.population)
            fitness_new = np.copy(self.fitness)
            
            for j in range(self.pop_size):
                # 随机选择记忆库中的索引 r_i
                r_i = random.randint(0, self.memory_size - 1)
                
                # 生成 CR
                # 此时 memory_cr[r_i] 可能是 0.5 或之前的成功值
                # 注意：SHADE 中如果 cr 接近 0 或 1 有特殊处理，这里沿用你之前的逻辑
                cr = np.clip(random.gauss(self.memory_cr[r_i], 0.1), 0, 1)
                
                # 生成 F (mut)
                while True:
                    mut = cauchy.rvs(loc=self.memory_f[r_i], scale=0.1)
                    if mut >= 1:
                        mut = 1
                        break
                    if mut > 0:
                        break
                
                # 随机选择 p 值 (pmin=2/NP, pmax=0.2)
                # 原代码逻辑：random.randint(int(0.05 * popsize), int(0.2 * popsize))
                # 这种随机 p 的策略也是 SHADE 的一种变体
                p_min = 2 / self.pop_size
                p_val = random.uniform(p_min, 0.2)
                top_p_count = max(1, int(p_val * self.pop_size))
                p_best_idx = random.randint(0, top_p_count - 1)
                x_best_p = self.population[p_best_idx]
                
                # 执行策略
                trial = self.strategy_current_to_pbest_1_bin_with_archive(
                    j, x_best_p, mut, cr, self.archive
                )
                
                f_trial = self.obj_func(trial)
                
                if f_trial < self.fitness[j]:
                    population_new[j] = trial
                    fitness_new[j] = f_trial
                    
                    success_cr.append(cr)
                    success_f.append(mut)
                    diff_fitness.append(abs(self.fitness[j] - f_trial))
                    
                    self.archive.append(self.population[j].copy())
                else:
                    population_new[j] = self.population[j]
                    fitness_new[j] = self.fitness[j]
            
            # 维护存档
            while len(self.archive) > self.pop_size:
                remove_idx = random.randint(0, len(self.archive) - 1)
                self.archive.pop(remove_idx)
                
            self.population = population_new
            self.fitness = fitness_new
            
            # 更新记忆库 (基于加权 Lehmer Mean)
            if len(success_cr) > 0:
                # 计算权重 (基于适应度提升量)
                total_diff = sum(diff_fitness)
                weights = [d / total_diff for d in diff_fitness]
                
                # 加权 Mean CR
                mean_cr_w = sum(w * s_cr for w, s_cr in zip(weights, success_cr))
                if self.memory_cr[self.k_idx] == -1 or mean_cr_w == 0:
                     self.memory_cr[self.k_idx] = -1 # 保持终端值或特殊处理
                else:
                     self.memory_cr[self.k_idx] = mean_cr_w
                
                # 加权 Lehmer Mean F
                sum_f_sq = sum(w * (f**2) for w, f in zip(weights, success_f))
                sum_f = sum(w * f for w, f in zip(weights, success_f))
                if sum_f > 0:
                    self.memory_f[self.k_idx] = sum_f_sq / sum_f
                
                # 移动指针
                self.k_idx = (self.k_idx + 1) % self.memory_size
                
            yield self.best_vector, self.fitness[0]

def shade_test_50(fun, bounds, iterations):
    result = []
    for num in range(50):
        optimizer = SHADE(fun, bounds, pop_size=100, iterations=iterations)
        it = list(optimizer.run())
        result.append(it[-1][-1])
        print(num, result[-1])
    print(f"Mean: {np.mean(result)}")

if __name__ == '__main__':
    from functions import fun_sphere
    temp_optimizer = SHADE(fun_sphere, [(-100, 100)] * 30, pop_size=100, iterations=1000)
    it = list(temp_optimizer.run())
    print(it[-1][1])