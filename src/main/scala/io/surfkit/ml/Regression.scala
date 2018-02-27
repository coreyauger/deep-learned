package io.surfkit.ml

import java.io.File
import java.nio.file.Paths
import io.surfkit.ml.Graph._
import org.nd4j.linalg.dataset.DataSet

/**
  * Created by suroot on 30/11/17.
  */
object Regression {

  def train(dir: File, index: Int, num: Int) = {

    val features = Input(Paths.get(dir.getAbsolutePath))


    println("----- Example Complete -----");
  }

}
