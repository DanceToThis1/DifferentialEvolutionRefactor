import numpy as np
import matplotlib.pyplot as plt
from paper.base import DifferentialEvolutionOptimizer

class JDE(DifferentialEvolutionOptimizer):
    """
    自适应参数差分进化算法 (jDE)
    """
    def __init__(self, obj_func, bounds, mut=0.5, cr=0.9, pop_size=100, iterations=1000):
        super().__init__(obj_func, bounds, pop_size, iterations)
        # 初始参数
        self.mut = mut 
        self.cr = cr

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            for j in range(self.pop_size):
                indexes = [index for index in range(self.pop_size) if index != j]
                a, b, c = self.population[np.random.choice(indexes, 3, replace=False)]
                
                # --- JDE 特有的参数自适应逻辑 ---
                # 注意：这里简化了原逻辑，严格的 jDE 应该为每个个体存储独立的 F 和 CR
                randj = np.random.rand(4)
                current_mut = self.mut
                current_cr = self.cr
                
                if randj[0] < 0.1:
                    current_mut = 0.1 + randj[1] * 0.9
                if randj[2] < 0.1:
                    current_cr = randj[3]
                # -----------------------------

                mutant = a + current_mut * (b - c)
                mutant = self.check_bounds(mutant)
                
                cross_points = np.random.rand(self.dimensions) < current_cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimensions)] = True
                trial = np.where(cross_points, mutant, self.population[j])
                
                f = self.obj_func(trial)
                if f < self.fitness[j]:
                    self.fitness[j] = f
                    self.population[j] = trial
                    if f < self.fitness[self.best_index]:
                        self.best_index = j
                        self.best_vector = trial
                        
            yield self.best_vector, self.fitness[self.best_index]

# 保留绘图和测试函数，内部调用新类
def jde_test(fun, bounds, mut=0.9, cr=0.1, iterations=3000, log=0, pop_size=100):
    optimizer = JDE(fun, bounds, mut, cr, pop_size, iterations)
    it = list(optimizer.run())
    print(it[-1])
    
    # 绘图逻辑
    x, f = zip(*it)
    plt.plot(f, label='jde')
    if log == 1:
        plt.yscale('log')
    plt.legend()
    plt.show()

def jde_test_50(fun, bounds, iterations):
    result = []
    for num in range(50):
        optimizer = JDE(fun, bounds, pop_size=100, iterations=iterations)
        it = list(optimizer.run())
        result.append(it[-1][-1])
        print(num, result[-1])
    
    # 数据保存逻辑先注释掉，后续统一处理
    mean_result = np.mean(result)
    print(f"Mean result: {mean_result}")