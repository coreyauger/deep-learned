package io.surfkit.ml

import java.nio.file.Files

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.nio.file.Path

object Graph {
  import scala.collection.JavaConverters._

  trait Stage{
    def forward: INDArray
  }

  trait Input extends Stage{
  }

  case class CsvInput(dir: Path) extends Input{
    override def forward: INDArray = {
      Nd4j.create(Files.newDirectoryStream(dir).iterator().asScala.toList.map { x =>
        Files.readAllLines(x).asScala.toList.map(_.toDouble).toArray
      }.toArray)
    }
  }

  case class RawInput(forward: INDArray) extends Input

  object Input{
    def apply(dir: Path) = CsvInput(dir)
    def apply(input: INDArray) = RawInput(input)
  }

  case class DenseLayer(units: Int, outputs: Int)(A: Stage) extends Stage{
    var W = (0 until units).map { _ => Nd4j.rand(outputs, A.forward.getRow(0).length()) }
    var b = (0 until units).map { _ => Nd4j.zeros(outputs, 1) }

    def debug = {
      println(s"W: ${W.head.shapeInfoToString()}")
      println(s"b: ${b.head.shapeInfoToString()}")
    }

    def forward = {
      Nd4j.vstack( W.zip(b).map{
        case xs =>  xs._1.mmul(A.forward).add(xs._2)
      }.asJavaCollection)
    }
  }
}
