package io.surfkit.ml

import java.nio.file.Files

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.nio.file.Path

//https://www.coursera.org/learn/multivariate-calculus-machine-learning/quiz/NrGhK/training-neural-networks
object Model {
  import scala.collection.JavaConverters._

  sealed trait Ml

  trait Layer extends Ml{
    def forward: INDArray
    def back(Aprev: INDArray): INDArray
  }

  trait Activation extends Ml{
    def derivative(Z: INDArray): INDArray
    def apply(Z: INDArray): INDArray
  }

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
    override def back(Aprev: INDArray): INDArray = Aprev
  }

  case class RawInput() extends Input{
    override def forward: INDArray = input
    override def back(Aprev: INDArray): INDArray = Aprev

  }

  object Input{
    def apply(dir: Path) = CsvInput(dir)
    def apply(input: INDArray) = RawInput()(input)
  }

  case class Dense(units: Int, activation: Activation, name: Option[String] = None)(Alayer: Layer) extends Layer{
    var W: INDArray = null
    var b: INDArray = null
    var A: INDArray = null
    var Z: INDArray = null

    def debug = {
      println(s"DENSE[${name.getOrElse("")}]")
      println(s"A: ${A.shapeInfoToString()}")
      println(s"W: ${W.shapeInfoToString()}")
      println(s"b: ${b.shapeInfoToString()}")
    }

    def forward = {
      A = Alayer.forward
      if(W == null){
        W = Nd4j.rand( units, A.shape()(1) )
        b = Nd4j.zeros( units, 1 )
      }
      println(s"DENSE[${name.getOrElse("")}]")
      println(s"A: ${A.shapeInfoToString()}")
      println(s"W: ${W.shapeInfoToString()}")
      Z = W.mmul(A.transpose()).addColumnVector(b)
      println(s"Z: ${Z.shapeInfoToString()}")
      activation(Z).transpose()
    }

    def back(Eo: INDArray) = {
      println(s"back for Dense[${name.getOrElse("")}]")
      //J = (J.T @ W3).T
      println(s"Eo: ${Eo.shapeInfoToString()}")
      println(s"W: ${W.shapeInfoToString()}")

      val Eh1 = Eo.mmul(W).transpose()
      println(s"Eh1: ${Eh1}")
      val Eh = Eh1.mmul(activation.derivative(Z).transpose())
      println(s"Eh: ${Eh}")
      val dWo = Eh.transpose().mmul(A.transpose())
      println(s"dWo: ${dWo}")

      // TODO: where to get the lr from ?
      val lr = 0.01

      val step = dWo.mul(lr)
      println(s"step: ${step}")
      W = W.sub( step )
      println(s"W: ${W}")

      // calculate this JW and this Jb
      //J = J @ a2.T / x.size

      //J = np.sum(J, axis=1, keepdims=True) / x.size
      Alayer.back(Eh)
    }
  }

  case class ReLu(name: Option[String] = None) extends Activation{
    def apply(Z: INDArray) = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear(Z))
    def derivative(Z: INDArray) = ???
  }


  case class Sigmoid(name: Option[String] = None) extends Activation{
    def apply(Z: INDArray) = {
      val A = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Sigmoid(Z))
      println(s"Sigmoid[${name.getOrElse("")}] activation: ${A}")
      A
    }
    //np.cosh(z/2)**(-2) / 4
    def derivative(Z: INDArray) = {
      val cosh = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Cosh(Z.div(2.0)))
      val cosh2 = cosh.mul(cosh)
      cosh2.div(4)
    }
  }

  case class Tanh(name: Option[String] = None) extends Activation{
    // compute the activation..
    def apply(Z: INDArray) = {
      val A = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Tanh(Z))
      println(s"Tanh[${name.getOrElse("")}] activation: ${A}")
      A
    }
    // der tanh: 1/np.cosh(z)**2
    def derivative(Z: INDArray) = {
      val cosh = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Cosh(Z))
      val cosh2 = cosh.mul(cosh)
      cosh2.rdiv(1.0)
    }
  }

  case class Identity(name: Option[String] = None) extends Activation{
    def apply(Z: INDArray) = {
      println(s"Identity[${name.getOrElse("")}] activation: ${Z}")
      Z
    }
    def derivative(Z: INDArray): INDArray = Z
  }



  trait LossFunction extends Ml{
    def apply(YHat: INDArray, Y: INDArray): INDArray
  }

  case object SquaredError extends LossFunction {
    override def apply(YHat: INDArray, Y: INDArray): INDArray = {
      println(s"YHat: ${YHat.shapeInfoToString()}")
      println(s"Y: ${Y.shapeInfoToString()}")
      val res = YHat.subi(Y)
      val diff = Nd4j.norm1( res )
      val diff2 = diff.mmul(diff)
      diff2.div(YHat.shape()(0))
    }
  }

  case object CrossEntropy extends LossFunction{
    override def apply(YHat: INDArray, Y: INDArray): INDArray = {
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

  class Model(inputs: Seq[Input], output: Layer, loss: LossFunction = SquaredError){
    def train(batch: INDArray, Y: INDArray) = {
      inputs.map(_(batch))
      val YHat = output.forward
      val l = loss(YHat, Y)
      println(s"Y: ${Y}")
      println(s"out: ${YHat}")
      println(s"loss: ${l}")

      output.back(YHat)
    }
  }
}
