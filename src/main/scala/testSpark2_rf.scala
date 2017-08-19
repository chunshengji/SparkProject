/*
* https://www.kaggle.com/c/santander-customer-satisfaction
* By Chunsheng Ji, 8/8/2017
* */
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.SparkSession
//import org.scalatest.FunSuite

object testSpark2_rf {
  def main(args:Array[String]):Unit = {
    val spark = SparkSession
      .builder().master("local")
      .appName("SparkMLSample")
      .getOrCreate()

    val df = spark.read.option("header", "true").option("inferSchema", "true")
      .csv("data/train.csv").toDF()

//    df.show(false)

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

//    training.show(false)

    val rf = new RandomForestClassifier()
      .setLabelCol("TARGET")
      .setFeaturesCol("scaledFeatures")
      .setNumTrees(10)
//    val lr = new LogisticRegression()
//      .setFeaturesCol("features")
////      .setFeaturesCol("scaledFeatures")
//      .setLabelCol("TARGET")
//      .setElasticNetParam(1.0)

    val dtModel = rf.fit(training)
    val df3 = dtModel.transform(test)

    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol(rf.getRawPredictionCol)
      .setLabelCol(rf.getLabelCol).setMetricName("areaUnderROC")

    println("areaUnderROC = " + evaluator.evaluate(df3))


  }
}
