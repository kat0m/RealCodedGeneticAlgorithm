import numpy as np
from config import *

M = np.loadtxt('input_data/M_D%d.txt' % INDIVIDUAL_LENGTH)
Oshift = np.loadtxt('input_data/shift_data.txt').flatten()

def Evaluation(chrom, func_num):
    if   func_num == 1:
        return sphere_function(chrom)
    elif func_num == 2:
        return ellips_function(chrom)
    elif func_num == 3:
        return bent_cigar_function(chrom)
    elif func_num == 4:
        return discus_function(chrom)
    elif func_num == 5:
        return different_powers_function(chrom)
    elif func_num == 6:
        return rosenbrock_function(chrom)
    elif func_num == 7:
        return schaffers_F7_function(chrom)
    elif func_num == 8:
        return ackley_function(chrom)
    elif func_num == 9:
        return weierstrass_function(chrom)
    elif func_num == 10:
        return griewank_function(chrom)
    elif func_num == 11:
        return rastrigin_function(chrom, r_flag = False)
    elif func_num == 12:
        return rastrigin_function(chrom, r_flag=True)
    elif func_num == 13:
        return step_rastrigin_function(chrom)
    elif func_num == 14:
        return schwefel_function(chrom, r_flag = False)
    elif func_num == 15:
        return schwefel_function(chrom, r_flag = True)
    elif func_num == 16:
        return katsuura_function(chrom)
    elif func_num == 17:
        return bi_rastrigin_function(chrom, r_flag = False)
    elif func_num == 18:
        return bi_rastrigin_function(chrom, r_flag = True)
    elif func_num == 19:
        return grie_rosen_function(chrom)
    elif func_num == 20:
        return escaffer_F6_function(chrom)
    elif func_num == 21:
        return cf01(chrom)
    elif func_num == 22:
        return cf02(chrom)
    elif func_num == 23:
        return cf03(chrom)
    elif func_num == 24:
        return cf04(chrom)
    elif func_num == 25:
        return cf05(chrom)
    elif func_num == 26:
        return cf06(chrom)
    elif func_num == 27:
        return cf07(chrom)
    elif func_num == 28:
        return cf08(chrom)
    else:
        print('test function does not exist.')
        return None

# ================================================================================
## 評価関数の定義
# Sphere Function
def sphere_function(x, shift_n=1, rotate_n=1, r_flag=False): # 1
    x = shiftfunc(x, shift_n)
    return np.sum(x ** 2)

# Rotated High Conditioned Elliptic Function
def ellips_function(x, shift_n=1, rotate_n=1, r_flag=True): # 2
    x = shiftfunc(x, shift_n)
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = oszfunc(x)
    r = np.arange(INDIVIDUAL_LENGTH) / (INDIVIDUAL_LENGTH - 1)
    return np.sum((1000000 ** r) * (x ** 2))
    
# Rotated Bent Cigar Function
def bent_cigar_function(x, shift_n=1, rotate_n=1, r_flag=True): # 3
    x = shiftfunc(x, shift_n)
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = asyfunc(x, 0.5)
    if r_flag:
        x = rotatefunc(x, rotate_n + 1)
    return 1000000 * np.sum(x ** 2) - (999999 * (x[0] ** 2))

# Rotated Discus Function
def discus_function(x, shift_n=1, rotate_n=1, r_flag=True): # 4
    x = shiftfunc(x, shift_n)
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = oszfunc(x)
    return np.sum(x ** 2) + 999999 * (x[0] ** 2)

# Different Powers Function
def different_powers_function(x, shift_n=1, rotate_n=1, r_flag=False): # 5
    x = shiftfunc(x, shift_n)
    if r_flag:
        x = rotatefunc(x, rotate_n)
    r = np.arange(INDIVIDUAL_LENGTH) / (INDIVIDUAL_LENGTH - 1)
    return np.sqrt(np.sum(np.abs(x) ** (2 + 4 * r)))

# Rotated Rosenbrock's Function
def rosenbrock_function(x, shift_n=1, rotate_n=1, r_flag=True): # 6
    x = shiftfunc(x, shift_n)
    x = x * 2.048/100 # shrink to the original search range
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = x + 1 # shift to origin
    f = 0
    for i in range(INDIVIDUAL_LENGTH - 1):
        f += 100 * (x[i] ** 2 - x[i+1]) ** 2 + (x[i] - 1.0) ** 2
    return f

# Rotated Schaffers F7 Function
def schaffers_F7_function(x, shift_n=1, rotate_n=1, r_flag=True): # 7
    x = shiftfunc(x, shift_n)
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = asyfunc(x, 0.5)
    if r_flag:
        x = rotatefunc(x, rotate_n + 1)
    x = x * (10 ** (np.arange(INDIVIDUAL_LENGTH) / (2 * (INDIVIDUAL_LENGTH - 1))))
    z = np.array([np.sqrt(x[i] ** 2 + x[i+1] ** 2) for i in range(INDIVIDUAL_LENGTH - 1)])
    f = 0
    for i in range(INDIVIDUAL_LENGTH - 1):
        f += (np.sqrt(z[i]) + np.sqrt(z[i]) * (np.sin(50 * z[i] ** 0.2)) ** 2) ** 2
    return f / (INDIVIDUAL_LENGTH - 1)

# Rotated Ackley's Function
def ackley_function(x, shift_n=1, rotate_n=1, r_flag=True): # 8
    x = shiftfunc(x, shift_n)
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = asyfunc(x, 0.5)
    if r_flag:
        x = rotatefunc(x, rotate_n + 1)
    x = x * (10 ** (np.arange(INDIVIDUAL_LENGTH) / (2 * (INDIVIDUAL_LENGTH - 1))))
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2)/INDIVIDUAL_LENGTH)) \
           -np.exp(np.sum(np.cos(2 * np.pi * x))/INDIVIDUAL_LENGTH) + 20 + np.e

# Rotated Weierstrass Function
def weierstrass_function(x, shift_n=1, rotate_n=1, r_flag=True): # 9
    a, b, k_max = 0.5, 3, 20
    x = shiftfunc(x, shift_n) / 200
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = asyfunc(x, 0.5)
    if r_flag:
        x = rotatefunc(x, rotate_n + 1)
    x = x * (10 ** (np.arange(INDIVIDUAL_LENGTH) / (2 * (INDIVIDUAL_LENGTH - 1))))
    f = 0
    k = np.arange(k_max + 1)
    for xi in x:
        f += np.sum((a ** k) * np.cos(2 * np.pi * (b ** k) * (xi + 0.5)))
    f -= INDIVIDUAL_LENGTH * np.sum((a ** k) * np.cos(np.pi * (b ** k)))
    return f

# Rotated Griewank's Function
def griewank_function(x, shift_n=1, rotate_n=1, r_flag=True): # 10
    x = shiftfunc(x, shift_n) * 6
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = x * (100 ** (np.arange(INDIVIDUAL_LENGTH) / (2 * (INDIVIDUAL_LENGTH - 1))))
    r = np.sqrt(np.arange(1, INDIVIDUAL_LENGTH + 1))
    return np.sum(x ** 2 / 4000) - np.prod(np.cos(x / r)) + 1

# (Rotated) Rastrigin's Function
def rastrigin_function(x, shift_n=1, rotate_n=1, r_flag=True): # 11, 12
    x = shiftfunc(x, shift_n) * 0.0512
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = oszfunc(x)
    x = asyfunc(x, 0.2)
    if r_flag:
        x = rotatefunc(x, rotate_n + 1)
    x = x * (10 ** (np.arange(INDIVIDUAL_LENGTH) / (2 * (INDIVIDUAL_LENGTH - 1))))
    if r_flag:
        x = rotatefunc(x, rotate_n)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * INDIVIDUAL_LENGTH

# Non-Continuous Rotated Rastrigin's Function
def step_rastrigin_function(x, shift_n=1, rotate_n=1, r_flag=True): # 13
    x = shiftfunc(x, shift_n) * 0.0512
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = np.where(np.abs(x) <= 0.5, x, np.round(2 * x)/2)
    x = oszfunc(x)
    x = asyfunc(x, 0.2)
    if r_flag:
        x = rotatefunc(x, rotate_n + 1)
    x = x * (10 ** (np.arange(INDIVIDUAL_LENGTH) / (2 * (INDIVIDUAL_LENGTH - 1))))
    if r_flag:
        x = rotatefunc(x, rotate_n)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * INDIVIDUAL_LENGTH

# (Rotated) Schwefel's Function
def schwefel_function(x, shift_n=1, rotate_n=1, r_flag=False): # 14, 15
    x = shiftfunc(x, shift_n) * 10
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = x * (10 ** (np.arange(INDIVIDUAL_LENGTH) / (2 * (INDIVIDUAL_LENGTH - 1)))) + 420.9687462275036
    for i in range(INDIVIDUAL_LENGTH):
        if np.abs(x[i]) <= 5:
            x[i] = x[i] * np.sin(np.abs(x[i]) ** 0.5)
        elif x[i] > 500:
            x[i] = (500 - np.mod(x[i], 500)) * np.sin(np.sqrt(np.abs(500 - np.mod(x[i], 500)))) \
                   - (x[i] - 500) ** 2 / (10000 * INDIVIDUAL_LENGTH)
        else:
            x[i] = (np.mod(np.abs(x[i]), 500) - 500) * np.sin(np.sqrt(np.abs(np.mod(np.abs(x[i]), 500) - 500))) \
                   - (x[i] + 500) ** 2 / (10000 * INDIVIDUAL_LENGTH)
    return 418.9829 * INDIVIDUAL_LENGTH - np.sum(x)

# 怪しい
# Rotated Katsuura Function
def katsuura_function(x, shift_n=1, rotate_n=1, r_flag=True): # 16
    x = shiftfunc(x, shift_n) * 0.05
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = x * (100 ** (np.arange(INDIVIDUAL_LENGTH) / (2 * (INDIVIDUAL_LENGTH - 1))))
    if r_flag:
        x = rotatefunc(x, rotate_n + 1)
    f = 1.0
    r = np.arange(1, 32+1)
    for i in range(INDIVIDUAL_LENGTH):
        f *= (1 + i * np.sum(np.abs((2 ** r) * x[i] - np.round((2 ** r) * x[i]))/(2 ** r))) ** (10/(INDIVIDUAL_LENGTH ** 1.2))
    return 10 / (INDIVIDUAL_LENGTH ** 2) * f - 10 / (INDIVIDUAL_LENGTH ** 2)

# (Rotated) Lunacek bi-Rastrigin Function
def bi_rastrigin_function(x, shift_n=1, rotate_n=1, r_flag=False): # 17, 18
    mu0 = 2.5
    s = 1 - 1 / (2 * np.sqrt(INDIVIDUAL_LENGTH + 20) - 8.2)
    d = 1
    mu1 = -np.sqrt((mu0 ** 2 - d)/s)
    y = shiftfunc(x, shift_n) * 0.1
    x = 2 * np.sign(Oshift[:INDIVIDUAL_LENGTH]) * y + mu0
    if r_flag:
        z = rotatefunc(x-mu0, 1)
    else:
        z = x - mu0
    z = z * (100 ** (np.arange(INDIVIDUAL_LENGTH) / (2 * (INDIVIDUAL_LENGTH - 1))))
    if r_flag:
        z = rotatefunc(z, 2)

    return min(np.sum((x-mu0) ** 2), d * INDIVIDUAL_LENGTH + s * np.sum(x-mu1) ** 2) \
           + 10 * (INDIVIDUAL_LENGTH - np.sum(np.cos(2 * np.pi * z)))

# Rotated Expanded Griewank's plus Rosenbrock's Function
def grie_rosen_function(x, shift_n=1, rotate_n=1, r_flag=True): # 19
    x = shiftfunc(x, shift_n) * 0.05
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = x + 1
    f = 0
    for i in range(INDIVIDUAL_LENGTH - 1):
        t1 = x[i] ** 2 - x[i+1]
        t2 = x[i] - 1.0
        tmp = 100 * (t1 ** 2) + t2 ** 2
        f += tmp ** 2 / 4000 - np.cos(tmp) + 1
    t1 = x[INDIVIDUAL_LENGTH - 1] ** 2 - x[0]
    t2 = x[INDIVIDUAL_LENGTH - 1] - 1.0
    tmp = 100 * (t1 ** 2) + t2 ** 2
    f += tmp ** 2 / 4000 - np.cos(tmp) + 1
    return f

# Expanded Scaffer’s F6 Function
def escaffer_F6_function(x, shift_n=1, rotate_n=1, r_flag=True): # 20
    x = shiftfunc(x, shift_n)
    if r_flag:
        x = rotatefunc(x, rotate_n)
    x = asyfunc(x, 0.5)
    if r_flag:
        x = rotatefunc(x, rotate_n + 1)
    f = 0
    for i in range(INDIVIDUAL_LENGTH - 1):
        t1 = (np.sin(np.sqrt(x[i] ** 2 + x[i+1] ** 2))) ** 2 - 0.5
        t2 = (1.0 + 0.001 * (x[i] ** 2 + x[i+1] ** 2)) ** 2
        f += 0.5 + t1 / t2
    t1 = (np.sin(np.sqrt(x[INDIVIDUAL_LENGTH - 1] ** 2 + x[0] ** 2))) ** 2 - 0.5
    t2 = (1.0 + 0.001 * (x[INDIVIDUAL_LENGTH - 1] ** 2 + x[0] ** 2)) ** 2
    f += 0.5 + t1 / t2
    return f

# Composition Function 1 (n=5, Rotated)
def cf01(x): # 21
    cf_num = 5
    delta = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    bias = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    fit = np.zeros(cf_num)
    fit[0] = rosenbrock_function(x, shift_n=1, rotate_n=1, r_flag=True)
    fit[1] = different_powers_function(x, shift_n=2, rotate_n=2, r_flag=True) * 1e-6
    fit[2] = bent_cigar_function(x, shift_n=3, rotate_n=3, r_flag=True) * 1e-26
    fit[3] = discus_function(x, shift_n=4, rotate_n=4, r_flag=True) * 1e-6
    fit[4] = sphere_function(x, shift_n=5, rotate_n=5, r_flag=False) * 0.1
    return cf_cal(x, cf_num, fit, delta, bias)

# Composition Function 2 (n=3, Unrotated)
def cf02(x): # 22
    cf_num = 3
    delta = np.array([20.0, 20.0, 20.0])
    bias = np.array([0.0, 100.0, 200.0])
    fit = np.zeros(cf_num)
    for i in range(cf_num):
        fit[i] = schwefel_function(x, shift_n=i+1, rotate_n=i+1, r_flag=False)
    return cf_cal(x, cf_num, fit, delta, bias)

# Composition Function 3 (n=3, Rotated)
def cf03(x): # 23
    cf_num = 3
    delta = np.array([20.0, 20.0, 20.0])
    bias = np.array([0.0, 100.0, 200.0])
    fit = np.zeros(cf_num)
    for i in range(cf_num):
        fit[i] = schwefel_function(x, shift_n=i+1, rotate_n=i+1, r_flag=True)
    return cf_cal(x, cf_num, fit, delta, bias)

# Composition Function 4 (n=3, Rotated)
def cf04(x): # 24
    cf_num = 3
    delta = np.array([20.0, 20.0, 20.0])
    bias = np.array([0.0, 100.0, 200.0])
    fit = np.zeros(cf_num)
    fit[0] = schwefel_function(x, shift_n=1, rotate_n=1, r_flag=True) * 0.25
    fit[1] = rastrigin_function(x, shift_n=2, rotate_n=2, r_flag=True)
    fit[2] = weierstrass_function(x, shift_n=3, rotate_n=3, r_flag=True) * 2.5
    return cf_cal(x, cf_num, fit, delta, bias)

# Composition Function 5 (n=3, Rotated)
def cf05(x): # 25
    cf_num = 3
    delta = np.array([10.0, 30.0, 50.0])
    bias = np.array([0.0, 100.0, 200.0])
    fit = np.zeros(cf_num)
    fit[0] = schwefel_function(x, shift_n=1, rotate_n=1, r_flag=True) * 0.25
    fit[1] = rastrigin_function(x, shift_n=2, rotate_n=2, r_flag=True)
    fit[2] = weierstrass_function(x, shift_n=3, rotate_n=3, r_flag=True) * 2.5
    return cf_cal(x, cf_num, fit, delta, bias)

# Composition Function 6 (n=5, Rotated)
def cf06(x): # 26
    cf_num = 5
    delta = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    bias = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    fit = np.zeros(cf_num)
    fit[0] = schwefel_function(x, shift_n=1, rotate_n=1, r_flag=True) * 0.25
    fit[1] = rastrigin_function(x, shift_n=2, rotate_n=2, r_flag=True)
    fit[2] = ellips_function(x, shift_n=3, rotate_n=3, r_flag=True) * 1e-7
    fit[3] = weierstrass_function(x, shift_n=4, rotate_n=4, r_flag=True) * 2.5
    fit[4] = griewank_function(x, shift_n=5, rotate_n=5, r_flag=True) * 10
    return cf_cal(x, cf_num, fit, delta, bias)

# Composition Function 7 (n=5, Rotated)
def cf07(x): # 27
    cf_num = 5
    delta = np.array([10.0, 10.0, 10.0, 20.0, 20.0])
    bias = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    fit = np.zeros(cf_num)
    fit[0] = griewank_function(x, shift_n=1, rotate_n=1, r_flag=True) * 100
    fit[1] = rastrigin_function(x, shift_n=2, rotate_n=2, r_flag=True) * 10
    fit[2] = schwefel_function(x, shift_n=3, rotate_n=3, r_flag=True) * 2.5
    fit[3] = weierstrass_function(x, shift_n=4, rotate_n=4, r_flag=True) * 25
    fit[4] = sphere_function(x, shift_n=5, rotate_n=5, r_flag=False) * 0.1
    return cf_cal(x, cf_num, fit, delta, bias)

# Composition Function 8 (n=5, Rotated)
def cf08(x): # 28
    cf_num = 5
    delta = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    bias = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    fit = np.zeros(cf_num)
    fit[0] = griewank_function(x, shift_n=1, rotate_n=1, r_flag=True) * 2.5
    fit[1] = rastrigin_function(x, shift_n=2, rotate_n=2, r_flag=True) * 2.5e-3
    fit[2] = schwefel_function(x, shift_n=3, rotate_n=3, r_flag=True) * 2.5
    fit[3] = weierstrass_function(x, shift_n=4, rotate_n=4, r_flag=True) * 5.0e-4
    fit[4] = sphere_function(x, shift_n=5, rotate_n=5, r_flag=False) * 0.1
    return cf_cal(x, cf_num, fit, delta, bias)


# ================================================================================
def shiftfunc(x, n=1):
    return x - Oshift[(n-1) * INDIVIDUAL_LENGTH : n * INDIVIDUAL_LENGTH]

def rotatefunc(x, n=1):
    return np.dot(x, M[(n-1) * INDIVIDUAL_LENGTH : n * INDIVIDUAL_LENGTH])

def asyfunc(x, beta):
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = x[i] ** (1 + beta * i / (INDIVIDUAL_LENGTH - 1) * np.sqrt(x[i]))
    return x

def oszfunc(x):
    xs = np.sign(x)
    c1 = np.where(x > 0, 10.0, 5.5)
    c2 = np.where(x > 0, 7.9, 3.1)
    x = np.where(x != 0, np.log(np.abs(x)), 0)
    return xs * np.exp(x + 0.049 * (np.sin(c1 * x) + np.sin(c2 * x)))

def cf_cal(x, cf_num, fit, delta, bias):
    len_x = len(x)
    w = np.zeros(cf_num)
    fit = fit + bias
    for i in range(cf_num):
        w[i] = 1 / np.sqrt(np.sum((x - Oshift[len_x * i : len_x * (i+1)]) ** 2)) \
               * np.exp(-np.sum((x - Oshift[len_x * i : len_x * (i+1)]) ** 2) / (2 * len_x * (delta[i] ** 2)))
    w = w / np.sum(w)
    return np.sum(w * fit)