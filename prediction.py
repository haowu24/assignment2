from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator

sc= SparkContext()
spark_session = SparkSession(sc)

input = spark_session.read.format("csv").load('file:/home/ec2-user/spark/ValidationDataset.csv', inferSchema='true',header = True ,sep =";")


feature = [a for a in input.columns if a != '""""quality"""""']

assembler = VectorAssembler(inputCols=feature,
                            outputCol="features")
data = assembler.transform(input)

model = LinearRegressionModel.load("/home/ec2-user/spark/lr.model")
Predictions  = model.transform(data)

eva = RegressionEvaluator(
    labelCol= '""""quality"""""', predictionCol="prediction", metricName="rmse")
f1 = eva.evaluate(Predictions)
print("f1 = "+str(f1))
