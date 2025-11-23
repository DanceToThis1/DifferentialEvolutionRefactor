import numpy as np
import pandas as pd
import random
import statistics
from paper.base import DifferentialEvolutionOptimizer

class SADE(DifferentialEvolutionOptimizer):
    """
    自适应差分进化算法 (Self-adaptive Differential Evolution)
    自动调整策略选择概率 (sp) 和参数 (cr)。
    """
    def __init__(self, obj_func, bounds, pop_size=20, iterations=1000, lp=5):
        super().__init__(obj_func, bounds, pop_size, iterations)
        self.lp = lp # 学习周期 (Learning Period)
        
        # 策略概率初始化
        self.strategies = [
            self.strategy_rand_1_bin,
            self.strategy_rand_to_best_2_bin,
            self.strategy_rand_2_bin,
            self.strategy_current_to_rand_1
        ]
        self.sp = np.array([0.25] * 4) # 初始每种策略概率相等
        
        # 记忆内存
        self.success_memory = np.zeros([self.lp, 4])
        self.failure_memory = np.zeros([self.lp, 4])
        self.cr_memory = [[], [], [], []] # 记录每种策略成功的 CR 值

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            
            # --- 学习阶段更新策略概率 (Learning Phase) ---
            if i >= self.lp: # 只有在过了初始学习期后才开始更新概率
                # 原始逻辑：基于过去 lp 代的成功/失败数计算 skg
                success_sum = np.sum(self.success_memory, axis=0)
                failure_sum = np.sum(self.failure_memory, axis=0)
                
                # 避免除以零，加一个极小值或 0.01 (原代码逻辑)
                skg = success_sum / (failure_sum + 0.01) + 0.01
                self.sp = skg / np.sum(skg)
            
            # 清空当前代的记忆槽位
            current_mem_idx = i % self.lp
            self.success_memory[current_mem_idx] = 0
            self.failure_memory[current_mem_idx] = 0

            for j in range(self.pop_size):
                # 1. 轮盘赌选择策略
                strategy_idx = np.random.choice(4, p=self.sp)
                
                # 2. 自适应参数生成
                # F (mut) 服从正态分布 N(0.5, 0.3)
                mut = random.gauss(0.5, 0.3)
                
                # CR 服从正态分布 N(median_CR, 0.1)
                if self.cr_memory[strategy_idx]:
                    cr_median = statistics.median(self.cr_memory[strategy_idx])
                else:
                    cr_median = 0.5 # 默认值
                
                cr = np.clip(random.gauss(cr_median, 0.1), 0, 1)
                
                # 3. 执行策略 (调用父类方法)
                if strategy_idx == 3: # current_to_rand_1 不需要 CR
                    trial = self.strategies[strategy_idx](j, mut)
                else:
                    trial = self.strategies[strategy_idx](j, mut, cr)
                
                # 4. 选择与反馈
                f = self.obj_func(trial)
                if f < self.fitness[j]:
                    self.fitness[j] = f
                    self.population[j] = trial
                    
                    # 成功：记录 CR 和 成功计数
                    self.cr_memory[strategy_idx].append(cr)
                    self.success_memory[current_mem_idx, strategy_idx] += 1
                    
                    if f < self.fitness[self.best_index]:
                        self.best_index = j
                        self.best_vector = trial
                else:
                    # 失败：记录失败计数
                    self.failure_memory[current_mem_idx, strategy_idx] += 1
            
            yield self.best_vector, self.fitness[self.best_index]

# 保持接口兼容
def sade_test_50(fun, bounds, iterations):
    result = []
    for num in range(50):
        optimizer = SADE(fun, bounds, pop_size=100, iterations=iterations)
        it = list(optimizer.run())
        result.append(it[-1][-1])
        print(num, result[-1])
    print(f"Mean result: {np.mean(result)}")