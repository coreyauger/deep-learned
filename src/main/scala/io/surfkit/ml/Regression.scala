package io.surfkit.ml

import java.io.File
import java.nio.file.Paths

import io.surfkit.ml.Model._
import io.surfkit.ml.utils.MathFunctions
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

/**
  * Created by suroot on 30/11/17.
  */
object Regression {

  def train(dir: File) = {

    val features = Input(Paths.get(dir.getAbsolutePath))

    val dense = Dense(units=128)(features)


    println("----- Example Complete -----")
  }


  def test = {
    val input = RawInput()
    val dense = Dense(units=128)(input)
    val activation = ReLu(None)(dense)
    val output = Dense(units=1)(activation)
    val out = Identity(None)(output)

    val model = new Model(inputs=Seq(input), output = out)

    val sumSquare = MathFunctions.simpleAddSquare(3, 2)
    println(sumSquare._1.shapeInfoToString())
    println(s"X: ${sumSquare._1}")
    println(s"Y: ${sumSquare._2}")

    model.train( sumSquare._1, sumSquare._2 )


    println("----- Example Complete -----")
  }

}
