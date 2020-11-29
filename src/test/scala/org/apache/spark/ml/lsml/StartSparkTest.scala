package org.apache.spark.ml.lsml

import org.apache.spark.sql.SparkSession
import org.scalatest.Ignore
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

@Ignore
class StartSparkTest extends  AnyFlatSpec with should.Matchers {

  "Spark" should "start context" in {
    val spark = SparkSession.builder
      .appName("Simple Application")
      .master("local[4]")
      .getOrCreate()

    Thread.sleep(60000)
  }

}
