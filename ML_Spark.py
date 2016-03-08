
# coding: utf-8

# In[1]:

#vérification du bon fonctionnement et du lancement du sc
#import des packages nécessaires
#sc
import utils
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pandas as pd
from pandas import DataFrame
from pyspark.mllib.stat import Statistics


# In[34]:

#chargement du fichier glass.csv
nomF = "glass"
dataGlass = sc.textFile("file:/C:/spark-1.6.0-bin-hadoop2.4/"+nomF+".csv")
#séparation des colonnes
lines = dataGlass.map(lambda line: utils.toRow(line, ';'))
#ligne 1 = header -> save header
nomColInit = lines.take(1)[0]
#save nb col
nbColInit = len(nomColInit)
#sruppression du header
parts = lines.filter(lambda line: nomColInit != line) 
parts = parts.filter(lambda line: len(line) == nbColInit)
#cptage nb lignes (sans le header)   
nbLignesInit = parts.count()
#cptage erreur : nb col header != nb col de la ligne 
partsError = parts.filter(lambda line: len(line) != nbColInit)
nbErrorL = partsError.count()
#tableau résumé 
data = pd.DataFrame({'1-nom Fichier':[nomF],
                       '2-nb Col':[nbColInit],
                       '3-nb Lignes': [nbLignesInit],
                        '4-nb Lignes error': [nbErrorL]})
data


# In[5]:

#La variable à prédire est le type : col=9
#nomColInit[9]

##################################################
# Répartition des classes de la variable à prédire

# traitement sur la 9ème col
typeV = parts.map(lambda line: line[9])
# map + reduce
distribType = typeV.map(lambda typeT: (typeT, 1)).reduceByKey(lambda a, b: a+b)
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


# In[27]:

##### En trichant #####
# Utilisation de pandas pour résumer les données + afficher la matrice de corrélation
df = pd.read_csv("file:/C:/spark-1.6.0-bin-hadoop2.4/"+nomF+".csv", sep = ";",header=0)
df.describe()
# Matrice de corrélation
# print(df.corr())


# In[37]:

# Utilisation du package mllib
# Basics Statistics
partsNum = parts.map(lambda line: line[0:8])
summary = Statistics.colStats(partsNum)
print(summary.mean())
print(summary.variance())
print(summary.numNonzeros())
Statistics.corr(partsNum, method="pearson")


# # Classification supervisée

# ## Naive Bayes

# In[3]:

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import utils_mesure
import pandas as pd

def parseLine(line):
    parts = line.split(';')
    label = float(parts[8])
    features = Vectors.dense([float(x) for x in parts[0:8]])
    return LabeledPoint(label, features)

data = sc.textFile("file:/C:/spark-1.6.0-bin-hadoop2.4/glass_svm.csv")

# suppression du header
nomColInit = data.first()
data2 = data.filter(lambda line: nomColInit != line) 
data = data2.map(parseLine)

# Echantillonnage 60% entrainement et 40% test
training, test = data.randomSplit([0.6, 0.4], seed=0)
# construction d'un modèle Bayesien sur l'échantillon d'entrainement
model = NaiveBayes.train(training, 1.0)

# Application sur l'echantillon test
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))

# Calcul des indicateurs du modèle
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v ).count() / test.count()
pG = 0
rG = 0
i = 1
while i < 7:
    a = utils_mesure.matriceConf(predictionAndLabel, i)
    pG = pG + utils_mesure.precision(a)
    rG = rG + utils_mesure.rappel(a)
    i += 1
pG = pG /6
rG = rG /6
fM = utils_mesure.f_mesure(rG, pG)

# Résumée des mesures du modèle
data = pd.DataFrame({'1-Méthode':["NaiveBayes"],
                    '2-Précision':[pG],
                       '3-Rappel':[rG],
                       '4-F_mesure': [fM]})
data



# # Classification non Supervisée

# ## Kmeans

# In[207]:

def parseLine(line):
    parts = line.split(';')
    features = array([float(x) for x in parts[0:9]])
    return features
def parseLine2(line):
    parts = line.split(';')
    features = array([float(x) for x in parts[0:8]])
    return features

data = sc.textFile("file:/C:/spark-1.6.0-bin-hadoop2.4/glass_svm.csv")

# suppression du header
nomColInit = data.first()
data2 = data.filter(lambda line: nomColInit != line) 
data = data2.map(parseLine)
dataTrain = data2.map(parseLine2)
# Construction du Kmeans
clusters = KMeans.train(dataTrain, 6, maxIterations=10,
        runs=10, initializationMode="random")
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = dataTrain.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))


# In[219]:

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


# ### Interprétation à finir
Avec Kmeans, 2 groupes se distinguent : 4 et 6
Le groupe gp1_P regroupe 123 des individus et mélange nettement gp1_R / gp2_R / gp3_R
# ## Gaussian Mixture 

# In[200]:

# Exemple Spark
from pyspark.mllib.clustering import GaussianMixture
from numpy import array

# Load and parse the data
data = sc.textFile("data/mllib/gmm_data.txt")
parsedData = data.map(lambda line: array([float(x) for x in line.strip().split(' ')]))

# Build the model (cluster the data)
gmm = GaussianMixture.train(parsedData, 2)

# output parameters of model
for i in range(2):
    print ("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
        "sigma = ", gmm.gaussians[i].sigma.toArray())


# # En cours
# 

# In[15]:

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
import utils_mesure
import pandas as pd
def parseLine(line):
    parts = line.split(';')
    label = float(parts[8])
    features = array([float(x) for x in parts[0:7]])
    return features
data = sc.textFile("file:/C:/spark-1.6.0-bin-hadoop2.4/glass_svm.csv")

# suppression du header
nomColInit = data.first()
data2 = data.filter(lambda line: nomColInit != line) 
data = data2.map(parseLine)
clusters = KMeans.train(data, 6, maxIterations=10,
        runs=10, initializationMode="random")
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))


# In[4]:

from pyspark.ml.classification import LogisticRegression

# Load training data
training = sqlContext.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))


# In[2]:

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


# In[1]:

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


# In[3]:

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


# In[2]:

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


# In[4]:

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
model2 = DecisionTree.trainClassifier(training, numClasses=6, categoricalFeaturesInfo={},
                                     impurity='entropy', maxDepth=5, maxBins=32)


# In[11]:

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

# Load and parse the data file into an RDD of LabeledPoint.
data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification tree model:')
print(model.toDebugString())

# Save and load model
#model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
#sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")

