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

object testSpark1 {
  def main(args:Array[String]):Unit = {
    val spark = SparkSession
      .builder().master("local")
      .appName("SparkMLSample")
      .getOrCreate()

    val trainSet = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/train.csv")
      .toDF()
    //    trainSet.show(2)
    //trainSet.select("label").groupBy(col("label")).count().sort(desc("count")).show()
//    val featureNames = trainSet.columns.drop(1).dropRight(2)
    val featureNames = trainSet.schema.fieldNames.filter(_!="ID").filter(_!="TARGET")

    //    println(featureNames.size)
    //  println(s"testcolNames is: ")
    //
    //    val testcolNames = test.columns

    val assembler = new VectorAssembler()
      .setInputCols(featureNames)
      .setOutputCol("features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(true)
      .setWithStd(true)

    val df1 = assembler.transform(trainSet)
    val scalerModel = scaler.fit(df1)
    val df2 = scalerModel.transform(df1)

    val Array(training, test) = df2.randomSplit(Array(0.6, 0.4))


    val lr = new LogisticRegression()
      //       .setRegParam(0.3)
      //       .setElasticNetParam(0.8)
      .setLabelCol("TARGET")
      .setFeaturesCol("scaledFeatures")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.0))
      .addGrid(lr.regParam, Array(0.0))
      .build()

    //    val paramGrid = new ParamGridBuilder()
    //      .addGrid(lr.elasticNetParam, Array(0.5))
    //      .addGrid(lr.regParam, Array(0.05, 0.01))
    //      .build()

    //    val model = pipeline.fit(training)

    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol(lr.getRawPredictionCol)
      .setLabelCol(lr.getLabelCol)
      .setMetricName("areaUnderROC")

    val cv = new CrossValidator()
      .setEstimator(lr)
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
//
//    val lrm: LogisticRegressionModel = cvModel
//      .bestModel.asInstanceOf[PipelineModel]
//      .stages
//      .last.asInstanceOf[LogisticRegressionModel]
//    println(s"Coefficients: ${lrm.coefficients} Intercept: ${lrm.intercept}")

    val bestresultTest = bestModel.transform(test)

    println("Area under ROC = " + evaluator.evaluate(bestresultTest))



  }
}
