import numpy as np
from paper.base import DifferentialEvolutionOptimizer

class JDE(DifferentialEvolutionOptimizer):
    """
    自适应参数差分进化算法 (jDE)
    """
    def __init__(self, obj_func, bounds, mut=0.5, cr=0.9, pop_size=100, iterations=1000):
        super().__init__(obj_func, bounds, pop_size, iterations)
        # 初始 F 和 CR
        self.mut = mut 
        self.cr = cr

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            for j in range(self.pop_size):
                indexes = [index for index in range(self.pop_size) if index != j]
                a, b, c = self.population[np.random.choice(indexes, 3, replace=False)]
                
                # --- JDE 特有的参数自适应逻辑 ---
                randj = np.random.rand(4)
                current_mut = self.mut
                current_cr = self.cr
                
                # 以 10% 的概率更新 F (mut)
                if randj[0] < 0.1:
                    current_mut = 0.1 + randj[1] * 0.9
                
                # 以 10% 的概率更新 CR
                if randj[2] < 0.1:
                    current_cr = randj[3]
                # -----------------------------

                # 变异
                mutant = a + current_mut * (b - c)
                mutant = self.check_bounds(mutant)
                
                # 交叉
                cross_points = np.random.rand(self.dimensions) < current_cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimensions)] = True
                trial = np.where(cross_points, mutant, self.population[j])
                
                # 选择
                f = self.obj_func(trial)
                if f < self.fitness[j]:
                    self.fitness[j] = f
                    self.population[j] = trial
                    # 如果个体更新成功，其实 JDE 应该更新该个体的 F 和 CR，
                    # 但在你原始代码中并未持久化存储每个个体的 F/CR，这里保持原逻辑。
                    if f < self.fitness[self.best_index]:
                        self.best_index = j
                        self.best_vector = trial
                        
            yield self.best_vector, self.fitness[self.best_index]