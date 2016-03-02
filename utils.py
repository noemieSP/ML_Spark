import matplotlib
matplotlib.use('agg')
import os
import unicodedata
import time
import csv
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *


#Lecture du fichier -> RDD
def linesFunction(loc, nomF, sc):
    spark_home = os.environ.get('SPARK_HOME', None)
    lines = sc.textFile(loc+nomF, minPartitions=500, use_unicode=False)
    return (lines)
	
#suppression des accents
def remove_accents(input_str):
    if type(input_str)!= str:
        nkfd_form = unicodedata.normalize('NFKD', input_str)
        return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])
    else:
        return input_str

def toRowSep(line,d):
    import csv
    for r in csv.reader([line], delimiter=d):
        return r
    
def toRow(line,sep):
    return toRowSep(line,sep)

def constField(listeType, listeColNom):
    nbColS = len(listeType)
    nbColNom = len(listeColNom)
    if nbColS != nbColNom:
        print('erreur!! le nb de colonne ne correspond pas')
        res = []
    else:
        i = 0
        res = []
        while i < nbColNom:
            if (listeType[i]=='str'):
                res = res + [StructField(listeColNom[i].replace('"',''), StringType(), True )]
            elif (listeType[i]=='number'):
                res = res + [StructField(listeColNom[i].replace('"',''), FloatType(), True )]
            elif (listeType[i]=='bin'):
                res = res + [StructField(listeColNom[i].replace('"',''), IntegerType(), True )]
            elif (listeType[i]=='date'):
                res = res + [StructField(listeColNom[i].replace('"',''), DateType(), True )]
            i += 1        
    return(res)
	
def tupleTrans2(y, defT):
    nbCol = len(y)
    nbT = len(defT)
    if (nbCol != nbT):
        res = [None]*nbT
        print('Erreur!! le nb de colonne ne correspond aux nbs de type indique')
    else:
        i = 0
        res = []
        while i < nbCol:
            if (y[i]=='""'):
                res = res + [None]
            else:
                    res = res + [str(y[i].replace('"',''))]
            i+=1
        res = tuple(res)
    return (res)
	
#Construction d'un schema pour requete SQL, avc possiblite d'echantillonnage
def constTableSQLstr(loc, nomF, sep, ech, perc, sc, sqlContext):
    lines = linesFunction(loc, nomF, sc)
    #Detection de la premiere ligne header
    lines = lines.map(lambda l: remove_accents(l))
    lines = lines.map(lambda line: toRow(line, sep))
    header = lines.take(1)[0]
    nbColInit = len(header)
            #supression de la ligne 
    parts = lines.filter(lambda line: header != line) 
    parts = parts.filter(lambda line: len(line) == nbColInit)
            #info nb lignes / nb col    
    nbLignesInit = parts.count()
    partsError = parts.filter(lambda line: len(line) != nbColInit)
    nbErrorL = partsError.count()
    if (nbErrorL!=0):
        parts = parts.filter(lambda line: len(line) == nbColInit)
    if (ech==True):
        parts = parts.sample(False, perc)
    nomColInit = lines.take(1)[0]
    listeTypeMax=['str']*nbColInit
    people = parts.map(lambda p:tupleTrans2(p, listeTypeMax))
        #schema de la table
    fields = constField(listeTypeMax, nomColInit)
    schema = StructType(fields)
        #creation de la table
    schemaPeople = sqlContext.createDataFrame(people, schema)
    return schemaPeople

	
#Affichage de la premiere ligne d'un fichier pour connaitre le nom des champs
def header(loc, nomF, sep, sc):
    spark_home = os.environ.get('SPARK_HOME', None)
    lines = sc.textFile(loc+nomF, use_unicode=False)
    lines = lines.map(lambda line: toRow(line, sep))
    header = lines.take(1)[0]
    return header