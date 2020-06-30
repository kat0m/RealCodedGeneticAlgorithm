## GA のパラメータ設定
# 遺伝子長(問題のサイズ)
INDIVIDUAL_LENGTH = 10
# 集団サイズ
POPULATION_SIZE = 100
# 最大評価回数
MAX_EVALUATION_NUM = 100 * POPULATION_SIZE
# 突然変異確率
MUTATION_RATE = 0.01
# 乱数のseed
SEED = 100

## 交叉の設定
# 使用する交叉の設定
SPX_flag = 0
BLX_flag = 1
if not SPX_flag ^ BLX_flag:
    BLX_flag, SPX_flag = 1, 0
# BLX-alphaの設定
ALPHA = 0.36


## Minimal Generation Gap Model の設定
NUM_CHILDREN = 2
NUM_SELECTION = 2

## 問題の設定
# 対象とする問題
#func_list = [i for i in range(1, 28+1)]
func_list = [ 1 ]
# 定義域(各遺伝子がとる値の範囲)
RANGE_MAX = 100
RANGE_MIN = -100
# 目的(MAX or MIN)
OPTIMIZE_TYPE = 'MIN'

## パスの設定
# 結果保存ディレクトリのパス
path_Results = 'Results/'
