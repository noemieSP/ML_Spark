import pandas as pd

#x est la prediction / v la classe reelle
def matriceConf(PredAndR, numClasse):
    VP = PredAndR.filter(lambda (x, v): x == v and x==numClasse).count()
    VN = PredAndR.filter(lambda (x, v): x == v and x!=numClasse).count()
    FP = PredAndR.filter(lambda (x, v): x == numClasse and v!=numClasse).count()
    FN = PredAndR.filter(lambda (x, v): x != numClasse and v==numClasse).count()
    #matrice de confusion: 
    data = pd.DataFrame({'1-1':[VP, FP],
                           '2-Non 1':[FN, VN]})
    return data

def precision(mConf):
    if mConf.ix[0][0] == 0:
        res = 0
    else:
        res = 1.0*mConf.ix[0][0] / (mConf.ix[0][0] + mConf.ix[1][0])
    return res

def rappel(mConf):
    if mConf.ix[0][0] == 0:
        res = 0
    else:
        res = 1.0*mConf.ix[0][0] / (mConf.ix[0][0] + mConf.ix[0][1])
    return res

def f_mesure(rappel, precision):
    res = 2.0*precision*rappel / precision+rappel
    return res