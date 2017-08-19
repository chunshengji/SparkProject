/*
* https://www.kaggle.com/c/santander-customer-satisfaction
* By Chunsheng Ji, 8/8/2017
* */
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions._
//import org.scalatest.FunSuite

import scala.collection.mutable

object testSpark {
  def main(args:Array[String]):Unit = {
    val spark = SparkSession
      .builder().master("local")
      .appName("SparkMLSample")
      .getOrCreate()

    val trainSet = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/train.csv")
      .withColumn("label", col("TARGET")*1.0).toDF()
//    trainSet.show(2)
    //trainSet.select("label").groupBy(col("label")).count().sort(desc("count")).show()

    val Array(training, test) = trainSet.randomSplit(Array(0.7, 0.3))

    val featureNames = training.columns.drop(1).dropRight(2)

    //    println(featureNames.size)
    //  println(s"testcolNames is: ")
//
//    val testcolNames = test.columns

    val assembler = new VectorAssembler()
      .setInputCols(featureNames)
      .setOutputCol("features")
//    training.show(1)
    // try StandardScaler

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(true)
      .setWithStd(true)

    val lr = new LogisticRegression()
      .setMaxIter(10)
//       .setRegParam(0.3)
//       .setElasticNetParam(0.8)
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.5,1.0))
      .addGrid(lr.regParam, Array(0.05, 0.01))
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(assembler, scaler, lr))
//    val paramGrid = new ParamGridBuilder()
//      .addGrid(lr.elasticNetParam, Array(0.5))
//      .addGrid(lr.regParam, Array(0.05, 0.01))
//      .build()

//    val model = pipeline.fit(training)

    val evaluator = new BinaryClassificationEvaluator()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

//    val model = trainValidationSplit.fit(training)

//    val lrModel = lr.fit(training)
//
    val cvModel = cv.fit(training)

//    println(s"Coefficients: ${cvModel.coefficients} Intercept: ${cvModel.intercept}")

    //
    val bestModel = cvModel.bestModel

    val lrm: LogisticRegressionModel = cvModel
      .bestModel.asInstanceOf[PipelineModel]
      .stages
      .last.asInstanceOf[LogisticRegressionModel]
    println(s"Coefficients: ${lrm.coefficients} Intercept: ${lrm.intercept}")

    val bestresultTest = bestModel.transform(test)

    val bestEstimatorParamMap = cvModel
      .getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)
      ._1
    //  println(s"The parameter combinator for the best logistic regression model is: \n$bestEstimatorParamMap")

    val predictionsAndLabels = bestresultTest
      .select("prediction", "label")
      .rdd
      .map(row => (row.getDouble(0), row.getDouble(1)))
//
    val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
//
    val roc = metrics.roc

    val auROC = metrics.areaUnderROC

    println("Area under ROC = " + auROC)

    val metrics1 = new MulticlassMetrics(predictionsAndLabels)
    val confusionMatrix = metrics1.confusionMatrix
    println("Confusion Matrix: \n" + confusionMatrix.toString())

  }
}
