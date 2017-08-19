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

object testSpark2 {
  def main(args:Array[String]):Unit = {
    val spark = SparkSession
      .builder().master("local")
      .appName("SparkMLSample")
      .getOrCreate()

    val df = spark.read.option("header", "true").option("inferSchema", "true")
      .csv("data/train.csv").toDF()

    df.show(false)

    val featureColumns = df.schema.fieldNames.filter(_ != "ID").filter(_ != "TARGET")

    val ass = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val df1 = ass.transform(df)

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(true)
      .setWithStd(true)

    val scalerModel = scaler.fit(df1)
    val df2 = scalerModel.transform(df1)

    val Array(training, test) = df2.randomSplit(Array(0.6, 0.4))

    training.show(false)

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
//      .setFeaturesCol("scaledFeatures")
      .setLabelCol("TARGET")
      .setElasticNetParam(1.0)

    val lrModel = lr.fit(training)
    val df3 = lrModel.transform(test)

    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol(lr.getRawPredictionCol)
      .setLabelCol(lr.getLabelCol).setMetricName("areaUnderROC")

    println("areaUnderROC = " + evaluator.evaluate(df3))


  }
}
