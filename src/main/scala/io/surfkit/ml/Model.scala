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

  trait Activation extends Layer{
    def derivitave: INDArray
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

  case class Dense(units: Int, name: Option[String] = None)(Alayer: Layer) extends Layer{
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
        W = Nd4j.rand( units, A.getColumn(0).length() )
        b = Nd4j.zeros( units, 1 )
      }
      println(s"DENSE[${name.getOrElse("")}]")
      println(s"A: ${A}")
      Z = W.mmul(A).addColumnVector(b)
      println(s"Z: ${Z}")
      Z
    }

    def back(Eo: INDArray) = {
      println(s"back for Dense[${name.getOrElse("")}]")
      //J = (J.T @ W3).T
      println(s"Eo: ${Eo.shapeInfoToString()}")
      println(s"W: ${W.shapeInfoToString()}")

      val Eh = Eo.transpose().mmul(W).transpose()
      println(s"Eh: ${Eh}")
      val dWo = Eh.mmul(A)
      println(s"CWo: ${dWo}")

      // TODO: where to get the lr from ?
      val lr = 0.01

      W = W.sub( dWo.mul(lr) )

      // calculate this JW and this Jb
      //J = J @ a2.T / x.size

      //J = np.sum(J, axis=1, keepdims=True) / x.size
      Alayer.back(Eh)
    }
  }

  case class ReLu(name: Option[String] = None)(Zlayer: Layer) extends Activation{
    var Z: INDArray = null
    def forward = {
      Z = Zlayer.forward
      Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear(Z))
    }
    def derivitave = ???
    def back(Aprev: INDArray) = ???
  }


  case class Sigmoid(name: Option[String] = None)(Zlayer: Layer) extends Activation{
    var Z: INDArray = null
    def forward = {
      Z = Zlayer.forward
      Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Sigmoid(Z))
    }
    //np.cosh(z/2)**(-2) / 4
    def derivitave = {
      val cosh = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Cosh(Z.div(2.0)))
      val cosh2 = cosh.mul(cosh)
      cosh2.div(4)
    }
    //Jacobian
    def back(Eo: INDArray) = {
      println(s"back for Sigmoid[${name.getOrElse("")}]")
      val EoDSig = Eo.muli(derivitave)
      Zlayer.back(EoDSig)
    }
  }


  case class Tanh(name: Option[String] = None)(Zlayer: Layer) extends Activation{
    //var A: INDArray = null
    var Z: INDArray = null

    // compute the activation..
    def forward = {
      Z = Zlayer.forward
      val A = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Tanh(Z))
      println(s"Tanh[${name.getOrElse("")}] activation: ${A}")
      A
    }
    // der tanh: 1/np.cosh(z)**2
    def derivitave = {
      val cosh = Nd4j.getExecutioner().execAndReturn(new org.nd4j.linalg.api.ops.impl.transforms.Cosh(Z))
      val cosh2 = cosh.mul(cosh)
      cosh2.rdiv(1.0)
    }
    //Jacobian
    def back(Aprev: INDArray) = {
      println(s"back for Tanh[${name.getOrElse("")}]")
      println(s"Aprev: ${Aprev.shapeInfoToString()}")
      println(s"Z: ${Z.shapeInfoToString()}")
      val J = Aprev.muli(derivitave)
      Zlayer.back(J)
    }
  }


  case class Identity(name: Option[String] = None)(Zlayer: Layer) extends Activation{
    var Z: INDArray = null
    def forward = {
      Z = Zlayer.forward
      println(s"Identity[${name.getOrElse("")}] activation: ${Z}")
      Z
    }
    override def derivitave: INDArray = Z
    def back(Aprev: INDArray) = Zlayer.back(Aprev)
  }

  trait LossFunction extends Ml{
    def calculate(YHat: INDArray, Y: INDArray): INDArray
  }

  case object SquaredError extends LossFunction {
    override def calculate(YHat: INDArray, Y: INDArray): INDArray = {
      val diff = Nd4j.norm1( YHat.subi(Y) )
      val diff2 = diff.mmul(diff)
      diff2.div(YHat.shape()(0))
    }
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

  class Model(inputs: Seq[Input], output: Layer, loss: LossFunction = SquaredError){
    def train(batch: INDArray, Y: INDArray) = {
      inputs.map(_(batch))
      val YHat = output.forward
      val l = loss.calculate(YHat, Y)
      output.back(l)

      println(s"Y: ${Y}")
      println(s"out: ${YHat}")
      println(s"loss: ${l}")
    }
  }
}
