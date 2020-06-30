from Genetic import Genetic
from config import func_list

def main():
    ## GAの実行
    for func_num in func_list:
        Genetic(func_num)

if __name__ == '__main__' :
    main()