import pandas as pd
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from numpy import array

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

# Tab resume des mesures d'un modele
def tabSum(pl, nbGroup, nomM):
    pG, rG = 0, 0
    i = 1
    while i < (nbGroup+1):
        a = matriceConf(pl, i)
        pG = pG + precision(a)
        rG = rG + rappel(a)
        i += 1
    pG = pG /nbGroup
    rG = rG /nbGroup
    fM = f_mesure(rG, pG)

    # Resumee des mesures du modele
    data = pd.DataFrame({'1-Methode':[nomM],
                        '2-Precision':[pG],
                           '3-Rappel':[rG],
                           '4-F_mesure': [fM]})
    return data

def parseLine(line):
    parts = line.split(';')
    label = float(parts[8])
    features = Vectors.dense([float(x) for x in parts[0:8]])
    return LabeledPoint(label, features)

def parseLine2(line):
    parts = line.split(';')
    features = array([float(x) for x in parts[0:9]])
    return features

def parseLine3(line):
    parts = line.split(';')
    features = array([float(x) for x in parts[0:8]])
    return features