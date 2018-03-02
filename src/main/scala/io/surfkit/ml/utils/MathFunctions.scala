package io.surfkit.ml.utils

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object MathFunctions {

  def simpleAddSquare(numFeatures: Int, batchSize: Int = 64):(INDArray, INDArray) = {
    val X = Nd4j.rand( numFeatures, batchSize).mul(10)
    val Y = Nd4j.sum(X.dup, 0).mulRowVector(Nd4j.sum(X.dup, 0))
    (X, Y)
  }
}
