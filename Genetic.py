import numpy as np
import random
import datetime
import time
import os
from config import *
from Individual import Individual
from Evaluation import Evaluation
dt_now = datetime.datetime.now()
path_Result = None
np.random.seed(SEED)
random.seed(SEED)
eval_count = 0
best = None
if BLX_flag:
    xop = 'BLX-'
elif SPX_flag:
    xop = 'SPX-'

# ================================================================================
def Genetic(func_num):
    start_time = time.time()
    global eval_count, best, path_Result
    path_Result = path_Results + xop + dt_now.strftime('%Y%m%d-%H%M%S/') + 'F%02d/' % func_num
    try:
        os.makedirs(path_Result)
    except FileExistsError:
        pass
    eval_count, best = 0, None
    
    # 結果記録用のファイル
    elite_file = open(path_Result + 'elite.csv', 'a')
    population_file = open(path_Result + 'population.csv', 'ab')

    # 初期化
    individual = Initialization(func_num)
    elite_file.write('%d, %f\n' % (eval_count, best))
    print('ObjectiveFunction: F%02d' % func_num)
    print('< 適合度計算回数 > %d < 最良適合度 > %f' % (eval_count, best))
    chm_list = [individual[i].chrom for i in range(POPULATION_SIZE)]
    np.savetxt(population_file, chm_list, delimiter=',', fmt='%f')

    while eval_count < MAX_EVALUATION_NUM:
        individual = MinimalGenerationGapModel(individual, func_num)

        if eval_count % POPULATION_SIZE == 0:
            elite_file.write('%d, %f\n' % (eval_count, best))
            chm_list = [individual[i].chrom for i in range(POPULATION_SIZE)]
            np.savetxt(population_file, chm_list, delimiter=',', fmt='%f')
    print('< 適合度計算回数 > %d < 最良適合度 > %16.10f' % (eval_count, best))

    elapsed_time = time.time() - start_time
    print('time: %f[sec]' % elapsed_time)
    with open(path_Result + 'time.txt', 'w') as f:
        f.write('time: %f[sec]' % elapsed_time)
    elite_file.close()
    population_file.close()


# ================================================================================
## 個体集団の初期化
def Initialization(func_num):
    global eval_count, best

    individual = [None] * POPULATION_SIZE
    for i in range(POPULATION_SIZE):
        chrom = (RANGE_MAX - RANGE_MIN) * np.random.rand(INDIVIDUAL_LENGTH) + RANGE_MIN
        individual[i] = Individual(chrom)
        individual[i].fitness = Evaluation(individual[i].chrom, func_num)
        eval_count += 1
        if i == 0:
            best = individual[i].fitness
        elif OPTIMIZE_TYPE == 'MIN' and best > individual[i].fitness:
            best = individual[i].fitness
        elif OPTIMIZE_TYPE == 'MAX' and best < individual[i].fitness:
            best = individual[i].fitness
        
    return individual


# ================================================================================
## Minimal Generation Gap (MGG) Model
def MinimalGenerationGapModel(individual, func_num, dep_set = None, indep_set = None):
    global eval_count, best

    if SPX_flag:
        num_parents_1st = INDIVIDUAL_LENGTH + 1
    elif BLX_flag:
        num_parents_1st = 2
    num_parents_2nd = 2

    ## 複製選択
    parent_idx = random.sample([i for i in range(POPULATION_SIZE)], num_parents_1st)
    parent = [individual[pidx] for pidx in parent_idx]
    parentchrom = np.zeros((num_parents_1st, INDIVIDUAL_LENGTH))
    for i in range(num_parents_1st):
        parentchrom[i] = parent[i].chrom

    ## 交叉
    child = [Individual(np.zeros(INDIVIDUAL_LENGTH)) for _ in range(NUM_CHILDREN)]
    ## Simplex交叉
    if SPX_flag:
        childchrom = Simplex(parentchrom, num_parents_1st)
        for i in range(NUM_CHILDREN):
            child[i].chrom = childchrom[i]
        parent_idx = random.sample(parent_idx, 2)
    ## BLX-alpha
    elif BLX_flag:
        childchrom = BLX_alpha(parentchrom)
        for i in range(NUM_CHILDREN):
            child[i].chrom = childchrom[i]

    ## 突然変異
    child = Mutation(child)

    ## 生存選択
    individual = Replacement(individual, parent_idx, child, func_num)

    return individual


# ================================================================================
## BLX-alpha
def BLX_alpha(pare_chrom): # NUM_PARENTS, NUM_CHILDREN = 2, 2
    length = len(pare_chrom[0])
    child_chrom = np.zeros((2, length))
    for gn in range(length):
        coeff1, coeff2 = np.random.rand(2)
        max_p = max(pare_chrom[0][gn], pare_chrom[1][gn])
        min_p = min(pare_chrom[0][gn], pare_chrom[1][gn])
        mn = min_p - ALPHA * (max_p - min_p)
        mx = max_p + ALPHA * (max_p - min_p)

        r = mn + coeff1 * (mx - mn)
        s = mn + coeff2 * (mx - mn)

        r = min(r, RANGE_MAX)
        r = max(r, RANGE_MIN)
        s = min(s, RANGE_MAX)
        s = max(s, RANGE_MIN)

        child_chrom[0][gn] = r
        child_chrom[1][gn] = s

    return child_chrom


# ================================================================================
## Simplex
def Simplex(pare_chrom, num_parents):
    length = len(pare_chrom[0])
    # 拡張率の推奨値
    eps = np.sqrt(length + 1)
    # 重心Gを求める
    G = np.mean(pare_chrom, axis=0)
    # x_k = G + eps * (Pk - G) (k = 0, ..., n)
    x = G + eps * (pare_chrom - G)
    child = [None] * NUM_CHILDREN
    for idx in range(NUM_CHILDREN):
        r = np.random.rand(length) ** (1 / np.arange(1, length + 1))
        C = np.zeros((num_parents, length))
        for i in range(1, num_parents):
            C[i] = r * (x[i-1] - x[i] + C[i-1])
        child[idx] = x[num_parents - 1] + C[num_parents - 1]
        child[idx] = np.where(child[idx] > RANGE_MAX, RANGE_MAX, child[idx])
        child[idx] = np.where(child[idx] < RANGE_MIN, RANGE_MIN, child[idx])
    return child

# ================================================================================
## 突然変異
def Mutation(child):
    ## Uniform Mutation
    for i in range(NUM_CHILDREN):
        for j in range(INDIVIDUAL_LENGTH):
            rand_num = np.random.rand()
            if rand_num <= MUTATION_RATE:
                add = 2.0 * np.random.rand() - 1.0
                low = child[i].chrom[j] - RANGE_MIN
                up = RANGE_MAX - child[i].chrom[j]

                if add < 0.0:
                    child[i].chrom[j] += low * add
                else:
                    child[i].chrom[j] += up * add
    return child


# ================================================================================
## 生存選択
def Replacement(individual, parent_idx, child, func_num):
    global eval_count, best
    num_family = len(parent_idx) + len(child)
    # 子個体の評価
    for i in range(NUM_CHILDREN):
        child[i].fitness = Evaluation(child[i].chrom, func_num)
        eval_count += 1
        if OPTIMIZE_TYPE == 'MIN' and best > child[i].fitness:
            best = child[i].fitness
            print('★< 適合度計算回数 > %d < 最良適合度 > %16.10f' % (eval_count, best))
        elif OPTIMIZE_TYPE == 'MAX' and best < child[i].fitness:
            best = child[i].fitness
            print('★< 適合度計算回数 > %d < 最良適合度 > %16.10f\n' % (eval_count, best))
    
    inds = [individual[pidx] for pidx in parent_idx] + child
    selected_ind = [None] * NUM_SELECTION

    if OPTIMIZE_TYPE == 'MIN':
        inds.sort(key = lambda ind: ind.fitness)
    else:
        inds.sort(key = lambda ind: ind.fitness, reverse=True)

    ## 最良個体を選択
    selected_ind[0] = inds[0]

    ## ルーレット法による選択
    fit = np.zeros(num_family)
    for i in range(num_family):
        fit[i] = inds[i].fitness
    # スケーリング
    v = np.abs(fit[0] - fit[num_family-1])
    if v == 0:
        v = 1
    if OPTIMIZE_TYPE == 'MIN':
        fit = v / (fit - fit[0] + 0.5 * v)
    else:
        fit = fit - fit[num_family-1] + 0.1 * v
    # 選択確率
    f_sum = np.sum(fit)
    slct_prb = fit / f_sum
    cumsum_prb = np.cumsum(slct_prb)
    cumsum_prb[num_family - 1] = 1.0
    for i in range(1, NUM_SELECTION): 
        rand_num = np.random.rand()
        for j in range(num_family):
            if rand_num <= cumsum_prb[j]:
                selected_ind[i] = inds[j]
                break
    # 個体の入れ替え
    for i in range(NUM_SELECTION):
        individual[parent_idx[i]] = selected_ind[i]
    return individual