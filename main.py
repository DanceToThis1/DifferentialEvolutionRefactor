from functions import fun_rastrigin, fun_rosenbrock, fun_sphere
from paper.de import DE
from paper.jde import JDE
from paper.sade import SADE
from paper.jade import JADE
from paper.shade import SHADE
from paper.code import CoDE
from paper.epsde import EPSDE

def run_test():
    # 设置通用参数
    bounds = [(-100, 100)] * 30
    pop_size = 100
    iterations = 1000
    func = fun_rastrigin

    print(f"开始测试 {func.__name__} ...")

    # 1. 测试 DE
    print("\n--- Running DE ---")
    de_opt = DE(func, bounds, pop_size=pop_size, iterations=iterations)
    print(f"DE Result: {list(de_opt.run())[-1][1]}")

    # 2. 测试 JADE
    print("\n--- Running JADE ---")
    jade_opt = JADE(func, bounds, pop_size=pop_size, iterations=iterations)
    print(f"JADE Result: {list(jade_opt.run())[-1][1]}")

    # 3. 测试 EPSDE
    print("\n--- Running EPSDE ---")
    epsde_opt = EPSDE(func, bounds, pop_size=pop_size, iterations=iterations)
    print(f"EPSDE Result: {list(epsde_opt.run())[-1][1]}")

if __name__ == '__main__':
    run_test()