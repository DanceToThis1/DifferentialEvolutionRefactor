import numpy as np
import random
from paper.base import DifferentialEvolutionOptimizer

class EPSDE(DifferentialEvolutionOptimizer):
    """
    EPSDE: Ensemble of Parameters and Strategies Differential Evolution
    个体保留成功的策略/参数组合，失败时重新从池中随机选择。
    """
    def __init__(self, obj_func, bounds, pop_size=20, iterations=1000):
        super().__init__(obj_func, bounds, pop_size, iterations)
        
        # 策略池
        self.strategies = [
            self.strategy_rand_1_bin,
            self.strategy_best_2_bin,
            self.strategy_current_to_rand_1
        ]
        # 参数池 (简化为固定池，EPSDE原文通常使用离散的参数池)
        self.pool_f = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.pool_cr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # 每个个体的状态：[策略索引, F值, CR值]
        self.configs = []

    def _get_random_config(self):
        """随机从池中抽取一个配置"""
        s_idx = random.randint(0, len(self.strategies) - 1)
        f_val = random.choice(self.pool_f)
        cr_val = random.choice(self.pool_cr)
        return {"strat": s_idx, "F": f_val, "CR": cr_val}

    def initialize_population(self):
        super().initialize_population()
        # 初始化时，给每个个体随机分配一个配置
        self.configs = [self._get_random_config() for _ in range(self.pop_size)]

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            for j in range(self.pop_size):
                # 获取当前个体 j 的配置
                cfg = self.configs[j]
                strat_idx = cfg["strat"]
                mut = cfg["F"]
                cr = cfg["CR"]
                
                # 执行策略
                if strat_idx == 2: # current-to-rand 只需要 mut
                    trial = self.strategies[strat_idx](j, mut)
                else:
                    trial = self.strategies[strat_idx](j, mut, cr)
                
                f_trial = self.obj_func(trial)
                
                if f_trial < self.fitness[j]:
                    self.fitness[j] = f_trial
                    self.population[j] = trial
                    # 成功：保留配置，不做任何修改 (Keep current config)
                    
                    if f_trial < self.fitness[self.best_index]:
                        self.best_index = j
                        self.best_vector = trial
                else:
                    # 失败：该个体下一代重新随机选择配置
                    self.configs[j] = self._get_random_config()
            
            yield self.best_vector, self.fitness[self.best_index]

if __name__ == '__main__':
    from functions import fun_sphere
    opt = EPSDE(fun_sphere, [(-100, 100)]*30, iterations=1000)
    print(list(opt.run())[-1][1])