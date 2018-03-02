package io.surfkit.ml

import java.nio.file.Files

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.nio.file.Path

object Model {
  import scala.collection.JavaConverters._

  trait Layer{
    def forward: INDArray
  }

  trait Activation extends Layer{}

  trait Input extends Layer{
    var input: INDArray = null
    def apply(input: INDArray) = this.input = input
  }

  case class CsvInput(dir: Path) extends Input{
    override def forward: INDArray = {
      Nd4j.create(Files.newDirectoryStream(dir).iterator().asScala.toList.map { x =>
        Files.readAllLines(x).asScala.toList.map(_.toDouble).toArray
      }.toArray)
    }
  }

  case class RawInput() extends Input{
    override def forward: INDArray = input
  }

  object Input{
    def apply(dir: Path) = CsvInput(dir)
    def apply(input: INDArray) = RawInput()(input)
  }

  case class Dense(units: Int, outputs: Int)(A: Layer) extends Layer{
    var W: INDArray = null
    var b: INDArray = null

    def debug = {
      println(s"A: ${A.forward.shapeInfoToString()}")
      println(s"W: ${W.shapeInfoToString()}")
      println(s"b: ${b.shapeInfoToString()}")
    }

    def forward = {
      if(W == null){
        W = Nd4j.rand(units, A.forward.getColumn(0).length())
        b = Nd4j.zeros(units, 1)
      }
      debug
      W.mmul(A.forward).add(b)
    }
  }


  case class ReLu(opts: Option[Int] = None)(Z: Layer) extends Activation{
    def forward = {
      // TODO:
      Z.forward
    }
  }


  class Model(inputs: Seq[Input], output: Layer){
    def train(batch: INDArray) = {
      inputs.map(_(batch))
      val out = output.forward

      println(s"out: ${out}")
    }
  }
}
