import numpy as np
from paper.base import DifferentialEvolutionOptimizer

class CoDE(DifferentialEvolutionOptimizer):
    """
    CoDE: Composite Differential Evolution
    每次产生 3 个试验向量，选择其中最好的一个与目标向量竞争。
    """
    def __init__(self, obj_func, bounds, pop_size=20, iterations=1000):
        super().__init__(obj_func, bounds, pop_size, iterations)

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            for j in range(self.pop_size):
                # CoDE 的核心：生成三个候选向量 (trial vectors)
                
                # 候选 1: rand/1/bin, F=1.0, Cr=0.1
                v1 = self.strategy_rand_1_bin(j, mut=1.0, cr=0.1)
                f1 = self.obj_func(v1)
                
                # 候选 2: rand/2/bin, F=1.0, Cr=0.9
                v2 = self.strategy_rand_2_bin(j, mut=1.0, cr=0.9)
                f2 = self.obj_func(v2)
                
                # 候选 3: current-to-rand/1, F=0.8, Cr=0.2 (策略里无显式 Cr，但作为参数传递保持一致性)
                v3 = self.strategy_current_to_rand_1(j, mut=0.8)
                f3 = self.obj_func(v3)
                
                # 在三个候选中选最好的
                candidates_f = [f1, f2, f3]
                candidates_v = [v1, v2, v3]
                best_trial_idx = np.argmin(candidates_f)
                
                trial = candidates_v[best_trial_idx]
                f_trial = candidates_f[best_trial_idx]
                
                # 与父代竞争
                if f_trial < self.fitness[j]:
                    self.fitness[j] = f_trial
                    self.population[j] = trial
                    if f_trial < self.fitness[self.best_index]:
                        self.best_index = j
                        self.best_vector = trial
            
            yield self.best_vector, self.fitness[self.best_index]

# 保持接口兼容
def code_test_50(fun, bounds, iterations):
    result = []
    for num in range(50):
        optimizer = CoDE(fun, bounds, pop_size=30, iterations=iterations) # CoDE通常pop_size不用太大
        it = list(optimizer.run())
        result.append(it[-1][-1])
        print(num, result[-1])
    print(f"Mean: {np.mean(result)}")