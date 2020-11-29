package org.apache.spark.ml.lsml

import java.io.File

import breeze.linalg.DenseVector
import com.google.common.io.Files
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should
import org.apache.spark.sql.functions._

import scala.collection.JavaConverters._
import scala.util.Random

class LinearRegressionTest extends  AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  val lr = 0.01
  val maxIter = 100
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val vectors: Seq[Vector] = LinearRegressionTest._vectors
  lazy val wValues: linalg.DenseVector = LinearRegressionTest._idealW
  lazy val bValue: Double = LinearRegressionTest._idealB
  //  lazy val learnVectors = LinearRegressionTest._learnVectors
  lazy val learnData = LinearRegressionTest._learnData

  "Model" should "calculate weighted sum" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      w = wValues,
      b = bValue
    ).setInputCol("features")
      .setOutputCol("features")

    val result: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))

    result.length should be(2)

    val a: Vector = result(0)

    result(0)(0) should be(vectors(0)(0) * wValues(0) + vectors(0)(1) * wValues(1) + vectors(0)(2) * wValues(2) + bValue +- delta)
    result(1)(0) should be(vectors(1)(0) * wValues(0) + vectors(1)(1) * wValues(1) + vectors(1)(2) * wValues(2) + bValue +- delta)
  }

  "Estimator" should "produce functional model" in {
    val estimator: LinearRegression = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("prediction")
      .setLabelCol("target")
      .setLearningRate(lr)
      .setMaxIter(maxIter)

    val assembler = new VectorAssembler()
      .setInputCols(Array("x", "y", "z"))
      .setOutputCol("features")

    val vectorized = assembler.transform(learnData)

    val model = estimator.fit(vectorized)
    model.w(0) should be(wValues(0) +- delta)
    model.w(1) should be(wValues(1) +- delta)
    model.w(2) should be(wValues(2) +- delta)

    model.b should be(bValue +- delta)
  }
}

object  LinearRegressionTest extends WithSpark {
  lazy val _vectors = Seq(
    Vectors.dense(1, 2, 3),
    Vectors.dense(-1, 0.5, 0)
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }

  lazy val _learnRDD = spark.sparkContext.parallelize(
    Seq.fill(100000){(Random.nextDouble, Random.nextDouble, Random.nextDouble)}
  )
  lazy val _idealW = Vectors.dense(1.5, 0.3, -0.7).toDense
  lazy val _idealB = 7.0
  lazy val _learnData: DataFrame = spark.createDataFrame(_learnRDD)
    .toDF("x", "y", "z")
    .withColumn("target",
      lit(_idealW(0)) * col("x")
          + lit(_idealW(1)) * col("y")
          + lit(_idealW(2)) * col("z") + _idealB)
}
