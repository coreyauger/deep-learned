package io.surfkit.ml

import java.nio.file.Files

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.nio.file.Path

object Model {
  import scala.collection.JavaConverters._

  sealed trait Ml

  trait Layer extends Ml{
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

  case class Dense(units: Int)(A: Layer) extends Layer{
    var W: INDArray = null
    var b: INDArray = null

    def debug = {
      println(s"A: ${A.forward.shapeInfoToString()}")
      println(s"W: ${W.shapeInfoToString()}")
      println(s"b: ${b.shapeInfoToString()}")
    }

    def forward = {
      if(W == null){
        W = Nd4j.rand( units, A.forward.getColumn(0).length() )
        b = Nd4j.zeros( units, 1 )
      }
      println(s"A: ${A.forward}")
      val Z = W.mmul(A.forward).addColumnVector(b)
      println(s"Z: ${Z}")
      Z
    }
  }

  case class ReLu(opts: Option[Int] = None)(Z: Layer) extends Activation{
    def forward = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear(Z.forward))
  }

  case class Sigmoid(opts: Option[Int] = None)(Z: Layer) extends Activation{
    def forward = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Sigmoid(Z.forward))
  }

  case class Tanh(opts: Option[Int] = None)(Z: Layer) extends Activation{
    def forward = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Tanh(Z.forward))
  }

  case class Identity(opts: Option[Int] = None)(Z: Layer) extends Activation{
    def forward = Z.forward
  }

  trait LossFunction extends Ml{
    def calculate(YHat: INDArray, Y: INDArray): INDArray
  }

  case object CrossEntropy extends LossFunction{
    override def calculate(YHat: INDArray, Y: INDArray): INDArray = {
      val m = Y.shape()(1)
      println(s"m: ${m}")
      //(-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
      val logYHat = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Log(YHat.dup))
      val log1MinusYHat =  Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Log(YHat.dup.sub(1)))
      Nd4j.sum(
        Y.dup
          .mmul(logYHat.transpose )
          .add(
            YHat.dup.sub(1).mmul(log1MinusYHat.transpose )
          ), 1
      ).mul( -1.0 / m.toDouble )
    }
  }

  class Model(inputs: Seq[Input], output: Layer, loss: LossFunction = CrossEntropy){
    def train(batch: INDArray, Y: INDArray) = {
      inputs.map(_(batch))
      val YHat = output.forward
      val l = loss.calculate(YHat, Y)

      println(s"out: ${YHat}")
      println(s"loss: ${l}")
    }
  }
}
