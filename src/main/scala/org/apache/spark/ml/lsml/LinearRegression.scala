package org.apache.spark.ml.lsml

import breeze.linalg.{*, DenseMatrix, DenseVector => BreezeDV}
import breeze.stats.mean
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.types.StructType

import scala.collection.mutable.ArrayBuffer

trait LinearRegressionParams extends HasInputCol with HasOutputCol with HasLabelCol {
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}


class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))


  private val batchSize = 50
  private val eps = 1e-4
  private var lr = 0.001
  def setLearningRate(value: Double): this.type = {
    lr = value
    this
  }

  private var maxIter = 1000
  def setMaxIter(value: Int): this.type = {
    maxIter = value
    this
  }

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()
    implicit val doubleEncoder: Encoder[Double] = ExpressionEncoder()

    val vectors: Dataset[(Vector, Double)] = dataset.select(
      dataset($(inputCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
      vectors.first()._1.size
    )

    var w = BreezeDV.ones[Double](dim)
    var b = 1.0
    var error = Double.MaxValue
    var i = 0

    while (i < maxIter && error > eps) {
      val (wSummary, bSummary) = vectors.rdd.mapPartitions((data: Iterator[(Vector, Double)]) => {
        val wGradSummarizer = new MultivariateOnlineSummarizer()
        val bGradSummarizer = new MultivariateOnlineSummarizer()

        data.grouped(batchSize).foreach((group: Seq[(Vector, Double)]) => {
          val (fBuffer, lBuffer) = group.map(x => (
            x._1.toArray.to[ArrayBuffer], Array(x._2).to[ArrayBuffer]
          )).reduce((x, y) => {
            (x._1 ++ y._1, x._2 ++ y._2)
          })

          val (fArray, lArray) = (fBuffer.toArray, lBuffer.toArray)
          val X = DenseMatrix.create(fArray.size / dim, dim, fArray, 0, dim, true)
          val y = BreezeDV(lArray)

          val yCur = (X * w) + b
          val delta = y - yCur
          val step = X(::, *) * delta

          wGradSummarizer.add(mllib.linalg.Vectors.fromBreeze(mean(step(::, *)).t))
          bGradSummarizer.add(mllib.linalg.Vectors.dense(mean(delta)))
        })

        Iterator((wGradSummarizer, bGradSummarizer))
      }).reduce((x, y) => {
        (x._1 merge y._1, x._2 merge y._2)
      })
      error = bSummary.mean(0)

      val wGrad = wSummary.mean.asBreeze.toDenseVector *:* (-2.0) * lr
      w -= wGrad
      val bGrad = (-2.0) * lr * error
      b -= bGrad
      i += 1
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(w).toDense, b))
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = ???

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

class LinearRegressionModel private[lsml](
                           override val uid: String,
                           val w: DenseVector,
                           val b: Double) extends Model[LinearRegressionModel] with LinearRegressionParams {

  private[lsml] def this(w: DenseVector, b: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), w, b)

  override def copy(extra: ParamMap): Nothing = ???

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x: Vector) => {
        Vectors.dense((x.asBreeze dot w.asBreeze) + b)
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = ???
}