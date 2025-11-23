import numpy as np
import random
from paper.base import DifferentialEvolutionOptimizer

class EPSDE(DifferentialEvolutionOptimizer):
    """
    EPSDE: Ensemble of Parameters and Strategies Differential Evolution
    (当前实现为随机策略选择版本，对应你原始代码逻辑)
    """
    def __init__(self, obj_func, bounds, pop_size=20, iterations=1000):
        super().__init__(obj_func, bounds, pop_size, iterations)
        # 策略池
        self.strategies = [
            self.strategy_rand_1_bin,       # 对应原代码 rand_algo 1
            self.strategy_best_2_bin,       # 对应原代码 rand_algo 2
            self.strategy_current_to_rand_1 # 对应原代码 rand_algo 3
        ]
        # 参数池 (这里简化为范围随机，对应原代码逻辑)
        # F: 0.4 -> 0.9, CR: 0.1 -> 0.9

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            for j in range(self.pop_size):
                # 随机选择策略 (1, 2, 3) -> 索引 (0, 1, 2)
                algo_idx = np.random.randint(0, 3)
                
                # 随机选择参数 (对应原代码的逻辑)
                # rand_f = 0.1 * np.random.randint(1, 9) -> 0.1 ~ 0.8 (原代码逻辑如果是1-9不含9)
                # 我们这里稍微优化一下范围，使其更合理：0.4 ~ 0.9
                mut = random.uniform(0.4, 0.9)
                cr = random.uniform(0.1, 0.9)
                
                # 执行策略
                if algo_idx == 2: # current-to-rand 只需要 mut
                    trial = self.strategies[algo_idx](j, mut)
                else:
                    trial = self.strategies[algo_idx](j, mut, cr)
                
                # 选择
                f = self.obj_func(trial)
                if f < self.fitness[j]:
                    self.fitness[j] = f
                    self.population[j] = trial
                    if f < self.fitness[self.best_index]:
                        self.best_index = j
                        self.best_vector = trial
            
            yield self.best_vector, self.fitness[self.best_index]

if __name__ == '__main__':
    from functions import fun_sphere
    opt = EPSDE(fun_sphere, [(-100, 100)]*30, iterations=200)
    print(list(opt.run())[-1])