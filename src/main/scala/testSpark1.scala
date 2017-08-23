/*
* https://www.kaggle.com/c/santander-customer-satisfaction
* By Chunsheng Ji, 8/8/2017
* */

/*
8/19/17: add pca https://spark.apache.org/docs/2.1.0/ml-features.html#pca; auc = 0.6407
          use pca alone，no weight balanced, no standardscaler; auc = 0.5954, .setK(3) is too low
          try .setK(300), auc = 0.787
          try .setK(200), auc = 0.7858
          try .setK(100), auc = 0.7332

8/17/17: try weight balanced, before auc = 0.7873, after auc = 0.7479
8/12/17: USE StandardScaler, best auc = 0.7873, regParam = 0.01, elasticNetParam = 0.0
 */
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{PCA, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.functions._
//import org.scalatest.FunSuite

import scala.collection.mutable

object testSpark1 {

  // try Jiang, Ming's wtbalance
  def balanceDataset(dataset: DataFrame): DataFrame = {

    // Re-balancing (weighting) of records to be used in the logistic loss objective function
    val numNegatives = dataset.filter(dataset("TARGET") === 0).count
    val datasetSize = dataset.count
    val delta = 0.03
    val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize + delta

    println("balancingRatio is： " + balancingRatio)

    val calculateWeights = udf { d: Double =>
      if (d == 0.0) {
        1 * balancingRatio
      }
      else {
        (1 * (1.0 - balancingRatio))
      }
    }

    val weightedDataset = dataset.withColumn("classWeightCol", calculateWeights(dataset("TARGET")))
    weightedDataset
  }

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

    val featureNames = trainSet.schema.fieldNames.filter(_!="ID").filter(_!="TARGET")
    // don't include "ID" and "TARGET" in features

    val assembler = new VectorAssembler()
      .setInputCols(featureNames)
      .setOutputCol("features")
    // assembler features

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(true)
      .setWithStd(true)
//    standardscaler

    val df1 = assembler.transform(trainSet)
    val scalerModel = scaler.fit(df1)
    val df2 = scalerModel.transform(df1)

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(300)
      .fit(df2)
//    add pca https://spark.apache.org/docs/2.1.0/ml-features.html#pca; auc = 0.6407， setK(3)
//    use pca alone，no weight balanced, no standardscaler; auc = 0.5954, .setK(3) is too low
//    try .setK(300), auc = 0.787
//    try .setK(200), auc = 0.7858
//    try .setK(100), auc = 0.7332

    val df3 = pca.transform(df2)
    //
    val Array(training, test) = df3.randomSplit(Array(0.6, 0.4))

    val wttraining = balanceDataset(training)

    val lr = new LogisticRegression()
      //       .setRegParam(0.3)
      //       .setElasticNetParam(0.8)
      .setLabelCol("TARGET")
      .setFeaturesCol("pcaFeatures")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam, Array(0.0))
      .addGrid(lr.regParam, Array(0.001))
      .build()
    //  regParam = elasticNetParam == 0.0, auc == 0.751
    //  regParam == 0.01, elasticNetParam == 0.0, auc = 0.7873
    //  regParam == 0.01, elsaticNetParam == 1.0, auc = 0.7663


    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol(lr.getRawPredictionCol)
      .setLabelCol(lr.getLabelCol)
      .setMetricName("areaUnderROC")

    val cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(wttraining)

    //    println(s"Coefficients: ${cvModel.coefficients} Intercept: ${cvModel.intercept}")

    //
    val bestModel = cvModel.bestModel

    val lrm: LogisticRegressionModel = cvModel
      .bestModel.asInstanceOf[LogisticRegressionModel]

    println(s"Coefficients: ${lrm.coefficients} Intercept: ${lrm.intercept}")

    //
    val bestEstimatorParamMap = cvModel
      .getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)
      ._1

    println(bestEstimatorParamMap)

    val bestresultTest = bestModel.transform(test)

    println("Area under ROC = " + evaluator.evaluate(bestresultTest))



  }
}
