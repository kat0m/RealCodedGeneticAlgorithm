class Individual:
    '''
    chrom...染色体(解候補データ)
    fitness...最適化対象となる問題に対しての解候補データの評価
    '''
    chrom = None
    fitness = None

    def __init__(self, chrom):
        self.chrom = chrom