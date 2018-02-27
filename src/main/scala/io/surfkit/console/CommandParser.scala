package io.surfkit.console

import java.io.File

object CommandParser {

  case class Mode(name: String)
  object Mode{
    val none = Mode("none")
    val train = Mode("train")
  }

  case class CmdConfig(file: File = new File("."), xyz: Boolean = false, model: String = "",
                       verbose: Boolean = false, debug: Boolean = false,
                       mode: Mode = Mode.none, files: Seq[File] = Seq(),
                       index: Int = 0, num: Int = 0, ml: String = "")

  private[this] val parser = new scopt.OptionParser[CmdConfig]("deeply-disturbed") {
    head("deeply-disturbed", "0.0.1-SNAPSHOT")

    opt[File]('f', "files").valueName("<file1>,<file2>...")
      .action( (x, c) => c.copy(file = x) )
      .text("input file")

    opt[Unit]("verbose").action( (_, c) =>
      c.copy(verbose = true) ).text("verbose is a flag")

    opt[Unit]("debug").hidden().action( (_, c) =>
      c.copy(debug = true) ).text("this option is hidden in the usage text")

    help("help").text("prints this usage text")
    note("this utility can be used to alter production data and apply patches.  you must first be ssh port forwarded in order to perform these tasks.\n")

    cmd("train").required().action( (_, c) => c.copy(mode = Mode.train) ).
      text("train is a command.").
      children(
        opt[File]("f").abbr("s").action( (x, c) =>
          c.copy(file = x) ).text("directory of csv training examples."),
        opt[Int]("i").abbr("i").action( (x, c) =>
          c.copy(index = x) ).text("Index to start training at"),
        opt[Int]("n").abbr("n").action( (x, c) =>
          c.copy(num = x) ).text("number of training examples to user"),
        opt[String]("m").abbr("m").action( (x, c) =>
          c.copy(model = x) ).text("model [rnn, reg]")
      )
  }


  def parse(args: Array[String]):Option[CmdConfig] =
  // parser.parse returns Option[C]
    parser.parse(args, CmdConfig())

}