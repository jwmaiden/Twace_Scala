import AssemblyKeys._

assemblySettings

name := "Twace"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.2.1" % "provided"

libraryDependencies += "org.apache.spark" % "spark-graphx_2.10" % "1.2.1" % "provided"
