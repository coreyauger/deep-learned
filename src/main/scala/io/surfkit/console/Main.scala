package io.surfkit.console

import java.io.File
import java.util.UUID
import io.surfkit.ml.Regression
import scala.collection.Iterator


object Main extends App {

  println("CONSOLE rarg !!!")
  Regression.buildXorNetwork

  /*@inline def defined(line: String) = {
    line != null && line.nonEmpty
  }
  Iterator.continually(scala.io.StdIn.readLine).takeWhile(defined(_)).foreach{line =>
    println("read " + line)
    CommandParser.parse(line.split(' ')).map { cmd =>
      cmd.mode match {
        case CommandParser.Mode.train =>
          val model = cmd.model.toLowerCase
          Regression.buildXorNetwork

        case x => println(s"Unknown command '${x}'.")
      }
    }
  }*/

}

