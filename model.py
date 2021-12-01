from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler


sc= SparkContext()
spark_session = SparkSession(sc)

input = spark_session.read.format("csv").load('file:/home/ec2-user/spark/TrainingDataset.csv', inferSchema='true',header = True ,sep =";")
input.printSchema()


feature = [a for a in input.columns if a != '""""quality"""""']




assembler = VectorAssembler(inputCols=feature,
                            outputCol="features")
data = assembler.transform(input)


lr = LinearRegression(maxIter=30, regParam=0.3, elasticNetParam=0.4, featuresCol="features", labelCol='""""quality"""""')

model = lr.fit(data)

model.save("/home/ec2-user/spark/lr.model")