package io.surfkit.ml

import java.io.File
import java.nio.file.Paths

import io.surfkit.ml.Model._
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

/**
  * Created by suroot on 30/11/17.
  */
object Regression {

  def train(dir: File) = {

    val features = Input(Paths.get(dir.getAbsolutePath))

    val dense = Dense(units=128, outputs = 1)(features)


    println("----- Example Complete -----")
  }


  def test = {
    val input = RawInput()
    val dense = Dense(units=128, outputs = 4)(input)
    val activation = ReLu(None)(dense)

    val model = new Model(inputs=Seq(input), output = activation)

    model.train(  Nd4j.create(Array(Array(2.0, 3.0, 4.0),Array(2.0, 3.0, 4.0),Array(2.0, 3.0, 4.0),Array(2.0, 3.0, 4.0))).transpose() )


    println("----- Example Complete -----")
  }

}
