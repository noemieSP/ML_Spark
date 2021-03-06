
# coding: utf-8

# # Découverte du package MLLIB 

# ## Analyse explo (rapide) 

# In[1]:

import utils
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from pandas import DataFrame


# In[2]:

# chargement du fichier glass.csv
nomF = "glass"
dataGlass = sc.textFile("file:/C:/spark-1.6.0-bin-hadoop2.4/"+nomF+".csv")
# séparation des colonnes
lines = dataGlass.map(lambda line: utils.toRow(line, ';'))
# ligne 1 = header -> save header
nomColInit = lines.take(1)[0]
# save nb col
nbColInit = len(nomColInit)
# sruppression du header
parts = lines.filter(lambda line: nomColInit != line) 
parts = parts.filter(lambda line: len(line) == nbColInit)
# cptage nb lignes (sans le header)   
nbLignesInit = parts.count()
# cptage erreur : nb col header != nb col de la ligne 
partsError = parts.filter(lambda line: len(line) != nbColInit)
nbErrorL = partsError.count()
# tableau résumé 
data = pd.DataFrame({'1-nom Fichier':[nomF],
                       '2-nb Col':[nbColInit],
                       '3-nb Lignes': [nbLignesInit],
                        '4-nb Lignes error': [nbErrorL]})
data


# Le fichier a été correctement importé (0 ligne d'erreur). 
# 
# Il décrit la composition de 214 échantillon de verre, définie sur 8 variables numériques et une variable qualitative de typage de l'échantillon. 

# ### Répartition des classes

# In[3]:

# La variable à prédire est le "type" 
typeV = parts.map(lambda line: line[8])
# map + reduce
distribType = typeV.map(lambda typeT: (typeT, 1)).reduceByKey(lambda a, b: a+b)
# Digramme de répartition
i = 0
name = []
data = []
while i <6:
    name = name + [distribType.collect()[i][0]]
    data = data +[int(distribType.collect()[i][1])]
    i+=1
plt.pie(data, labels=name, autopct='%1.1f%%', startangle=90, shadow=True)
plt.axis('equal')
plt.show()


# Ce jeu de données ne nécessite pas de manipulation directe, il n’y a en effet aucune valeur manquante ni aucune erreur détectée. En revanche, nous constatons une inégalité dans la répartition des types : les building_windows représentent plus de 68% de la base. Nous pouvons déjà nous attendre à des difficultés d'obtenir un modèle prédictif fiable.

# In[4]:

##### En trichant #####
# Utilisation de pandas pour résumer les données + afficher la matrice de corrélation
df = pd.read_csv("file:/C:/spark-1.6.0-bin-hadoop2.4/"+nomF+".csv", sep = ";",header=0)
df.describe()
# Matrice de corrélation
# print(df.corr())


# ### Mllib Statistics

# In[5]:

from pyspark.mllib.stat import Statistics
# Basics Statistics
partsNum = parts.map(lambda line: line[0:8])
summary = Statistics.colStats(partsNum)
print(summary.mean())
print(summary.variance())
print(summary.numNonzeros())
Statistics.corr(partsNum, method="pearson")


# # Classification supervisée

# ## Naive Bayes

# In[6]:

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
import utils_mesure
nomF_svm = "glass_svm"
data = sc.textFile("file:/C:/spark-1.6.0-bin-hadoop2.4/"+nomF_svm+".csv")

# suppression du header
nomColInit = data.first()
data2 = data.filter(lambda line: nomColInit != line) 
data = data2.map(utils_mesure.parseLine)

# Echantillonnage 60% entrainement et 40% test
training, test = data.randomSplit([0.6, 0.4], seed=0)
# construction d'un modèle Bayesien sur l'échantillon d'entrainement
model = NaiveBayes.train(training, 1.0)

# Application sur l'echantillon test
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))

# Calcul des indicateurs du modèle
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v ).count() / test.count()

utils_mesure.tabSum(predictionAndLabel, 7, 'Naive Bayes')



# ## Decision Tree

# In[8]:

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
import utils_mesure
data = sc.textFile("file:/C:/spark-1.6.0-bin-hadoop2.4/"+nomF_svm+".csv")

# suppression du header
nomColInit = data.first()
data2 = data.filter(lambda line: nomColInit != line) 
data = data2.map(utils_mesure.parseLine)

# Echantillonnage 60% entrainement et 40% test
training, test = data.randomSplit([0.6, 0.4], seed=0)
# Construction du modèle
model = DecisionTree.trainClassifier(training, numClasses=7, categoricalFeaturesInfo={},
                                     impurity='entropy', maxDepth=10, maxBins=32)
# Test 
predictions = model.predict(test.map(lambda x: x.features))
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(test.count())
print('Test Error = ' + str(testErr))
print('Learned classification tree model:')
print(model.toDebugString())


# In[8]:

# Mesures globales du mmodèle
utils_mesure.tabSum(labelsAndPredictions, 7, 'Decision Tree')


# L'affichage de l'arbre de décision n'est pas très lisible, il serait intéressant d'écrire un module qui permette de faire de la visu. Avec un taux d'erreur à 38,88% le modèle n'est pas très fiable.

# ## Méthodes non utilisables

# ### Gradient Boosted Tree

# Note: GBTs do not yet support multiclass classification. For multiclass problems, please use decision trees or Random Forests.
# 
# Cette méthode n'est donc pas appliquable à notre jeu de données.

# ### Survival regression

# L'analyse de survie repose souvent sur des séries temporelles de données longitudinales.
# 
# Notre Data Set ne correspond pas à ce type de modèle

# # Classification non Supervisée

# ## Kmeans

# In[10]:

from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
import numpy


data = sc.textFile("file:/C:/spark-1.6.0-bin-hadoop2.4/"+nomF_svm+".csv")

# suppression du header
nomColInit = data.first()
data2 = data.filter(lambda line: nomColInit != line) 
data = data2.map(utils_mesure.parseLine2)
dataTrain = data2.map(utils_mesure.parseLine3)
# Construction du Kmeans
clusters = KMeans.train(dataTrain, 6, maxIterations=10,
        runs=10, initializationMode="random")
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = dataTrain.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))


# In[11]:

# Répartition des groupes
gpKmeans = data.map(lambda p: (clusters.predict(p[0:8]),p[8]))
gpKmeansK = gpKmeans.map(lambda p: (str(p[0])+"_"+str(p[1]),1))
gpKmeansK2 = gpKmeansK.reduceByKey(lambda a, b: a+b)
listeR = gpKmeansK2.collect()
# Construction du tableau CD
# Découpage
listeF = []
i = 0
maxN = len(listeR)
while i < maxN:
    listeR2 = listeR[i][0].split("_") + [str(listeR[i][1])]
    listeF = listeF + [listeR2]
    i += 1
listeF
# construction de la liste
i = 0
l = numpy.array([[0]*6]*6)
while i < maxN:
    a = int(listeF[i][0])
    b = int(float(listeF[i][1])) -1
    l[a][b] = int(listeF[i][2])
    i += 1
# Affichage 
df = pd.DataFrame(l, index = ['gp1_P', 'gp2_P', 'gp3_P', 'gp4_P', 'gp5_P', 'gp6_P'],
                  columns = ['gp1_R', 'gp2_R', 'gp3_R', 'gp4_R', 'gp5_R', 'gp6_R'])
df


# ### Interprétation (à finir)
Avec Kmeans, 2 groupes se distinguent : 4 et 6
Le groupe gp1_P regroupe 123 des individus et mélange nettement gp1_R / gp2_R / gp3_R
# ## Gaussian Mixture 

# In[12]:

from pyspark.mllib.clustering import GaussianMixture

# Construction du model avc le mm dataTrain que Kmeans
gmm = GaussianMixture.train(dataTrain, 6)

# sortie des parameters du modele
for i in range(2):
    print ("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
        "sigma = ", gmm.gaussians[i].sigma.toArray())


# ### Interprétation (à finir)

# # Mesures d'évaluation (en cours)

# In[30]:

from pyspark.mllib.evaluation import MultilabelMetrics

# Instantiate metrics object
metrics = MultilabelMetrics(labelsAndPredictions)

# Summary stats
print("Recall = %s" % metrics.recall())
print("Precision = %s" % metrics.precision())
print("F1 measure = %s" % metrics.f1Measure())
print("Accuracy = %s" % metrics.accuracy)


# ("You must build Spark with Hive. Export 'SPARK_HIVE=true' and run build/sbt assembly", ...)

# # En cours
# 

# In[1]:

# PIC
from __future__ import print_function
from pyspark.mllib.clustering import PowerIterationClustering, PowerIterationClusteringModel

# Load and parse the data
data = sc.textFile("data/mllib/pic_data.txt")
similarities = data.map(lambda line: tuple([float(x) for x in line.split(' ')]))

# Cluster the data into two classes using PowerIterationClustering
model = PowerIterationClustering.train(similarities, 2, 10)

model.assignments().foreach(lambda x: print(str(x.id) + " -> " + str(x.cluster)))



# In[2]:

def printM(x):
    return (str(x.id) + " -> " + str(x.cluster))
    


# In[3]:

model.assignments().foreach(lambda x: printM(x))


# In[12]:

# Multilayer perceptron classifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load training data
data = sqlContext.read.format("libsvm")    .load("data/mllib/sample_multiclass_classification_data.txt")
# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]
# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [4, 5, 4, 3]
# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
# train the model
model = trainer.fit(train)
# compute precision on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="precision")
print("Precision:" + str(evaluator.evaluate(predictionAndLabels)))


# In[ ]:

from pyspark.ml.classification import LogisticRegression

# Load training data
training = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))


# In[ ]:

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load and parse the data file, converting it to a DataFrame.
data = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a GBT model.
gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)

# Chain indexers and GBT in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)  # summary only


# In[13]:

# Random Forest
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load and parse the data file, converting it to a DataFrame.
data = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only


# In[ ]:

# Classification et regression 
# Méthodes linéaires
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
nomF = "glass_svm"
# Load and parse the data
def parsePoint(line):
    values = [x for x in line.split(';')]
    return LabeledPoint(values[8], values[0:7])

data = sc.textFile("file:/C:/spark-1.6.0-bin-hadoop2.4/"+nomF+".csv")
data.first()
nomColInit = data.first()
#sruppression du header
data2 = data.filter(lambda line: nomColInit != line) 

parsedData = data2.map(parsePoint)

# Build the model
#model = SVMWithSGD.train(parsedData, iterations=100)

# Evaluating the model on training data
#labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
#trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
#print("Training Error = " + str(trainErr))

# Save and load model
#model.save(sc, "myModelPath")
#sameModel = SVMModel.load(sc, "myModelPath")


# In[ ]:

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("data/mllib/sample_svm_data.txt")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# Save and load model
model.save(sc, "myModelPath2")
sameModel = LogisticRegressionModel.load(sc, "myModelPath2")

