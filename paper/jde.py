import numpy as np
import random
from paper.base import DifferentialEvolutionOptimizer

class JDE(DifferentialEvolutionOptimizer):
    """
    jDE: Self-Adapting Control Parameters in Differential Evolution
    每个个体拥有独立的 F 和 CR，仅在进化成功时保留，否则重置或更新。
    """
    def __init__(self, obj_func, bounds, pop_size=100, iterations=1000, 
                 tau1=0.1, tau2=0.1, f_init=0.5, cr_init=0.9):
        super().__init__(obj_func, bounds, pop_size, iterations)
        self.tau1 = tau1
        self.tau2 = tau2
        # 初始化每个个体的 F 和 CR
        self.F = np.full(pop_size, f_init)
        self.CR = np.full(pop_size, cr_init)

    def run(self):
        self.initialize_population()
        
        for i in range(self.iterations):
            for j in range(self.pop_size):
                indexes = [index for index in range(self.pop_size) if index != j]
                a, b, c = self.population[np.random.choice(indexes, 3, replace=False)]
                
                # --- jDE 参数更新逻辑 (针对个体 j) ---
                # 临时变量，用于生成 trial
                f_j = self.F[j]
                cr_j = self.CR[j]
                
                # 以 tau1 的概率更新 F
                if random.random() < self.tau1:
                    f_j = 0.1 + random.random() * 0.9
                
                # 以 tau2 的概率更新 CR
                if random.random() < self.tau2:
                    cr_j = random.random()
                # ----------------------------------

                # 变异 (使用更新后的 f_j)
                mutant = a + f_j * (b - c)
                mutant = self.check_bounds(mutant)
                
                # 交叉 (使用更新后的 cr_j)
                cross_points = np.random.rand(self.dimensions) < cr_j
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimensions)] = True
                trial = np.where(cross_points, mutant, self.population[j])
                
                f_trial = self.obj_func(trial)
                
                if f_trial < self.fitness[j]:
                    self.fitness[j] = f_trial
                    self.population[j] = trial
                    # 只有成功了，才把刚才新的 F 和 CR 更新到记忆数组中
                    self.F[j] = f_j
                    self.CR[j] = cr_j
                    
                    if f_trial < self.fitness[self.best_index]:
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

if __name__ == '__main__':
    from functions import fun_rastrigin
    temp_optimizer = JDE(fun_rastrigin, [(-5.12, 5.12)] * 30, pop_size=100, iterations=1000)
    it = list(temp_optimizer.run())
    print(it[-1][1])