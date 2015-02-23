import java.io.File

import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.graphx._
import org.apache.spark.rdd._
import au.com.bytecode.opencsv.CSVParser
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
import org.apache.spark.mllib.regression.LassoWithSGD
import org.apache.spark.mllib.util.MLUtils

object Twace {

  def main(args: Array[String]) {  

      if (args.length != 3) {
      println("Usage: /path/to/spark/bin/spark-submit --class Twace " +
        "target/scala-*/twace-als-assembly-*.jar twitterUsersFile twitterConnectionsFile")
      sys.exit(1)
    }
	
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)	
	
    // set up environment
	
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
	
	// Pull in all scraped Twitter user records	
	val userRecords = sc.textFile(args(0)).mapPartitions ( lines => {
		val parser = new CSVParser(',')
		lines.map( line => {
			val columns = parser.parseLine(line)
			// format: (User Id, (User Name, Number of Followers, Location))
			(columns(0).toLong, (columns(1), columns(2).toLong, columns(3)))			
         })
	})
	
	// Pull in the scraped Retweet connections between users
	val userConnections = sc.textFile(args(1)).mapPartitions ( lines => {
		val parser = new CSVParser(',')
		lines.map( line => {
			val columns = parser.parseLine(line)
			// format: (Retweet User Id, (Original User Id, Tweet Id))
			(columns(0).toLong, (columns(1).toLong, columns(2).toLong))
         })
	})
	
	// Create a graph and show to which and from which locations Chase is getting retweets
	showRetweetImpressionsByLocation(userRecords, userConnections)					 
	
	// Pull in the retweet factors - number of retweets vs. specific hashtags, user mentions, etc
	val rtwt_data = MLUtils.loadLibSVMFile(sc, args(2))		
	
	val splits = rtwt_data.randomSplit(Array(0.8, 0.2))
	val (trainingData, testData) = (splits(0), splits(1))
	
	// Fit to linear models and write MSE value
	testLinearModels( trainingData, testData )
	
	// Fit to Random Forest model and write MSE value
	testRandomForest( trainingData, testData )
	
    // clean up
    sc.stop()
  }	
  
  def showRetweetImpressionsByLocation(userRecords : RDD[(Long, (String, Long, String))],  userConnections : RDD[(Long, (Long, Long))])
  {
		// Create a graph and show to which and from which locations Chase is getting retweets
				
		// Create edges from the user connection data
		val userConnectionEdges = userConnections.map( { case (rtwt_id, (orig_id, _)) => (rtwt_id, orig_id) -> 1} )
												 .reduceByKey(_ + _)
												 .map( { case ((rtwt_id, orig_id), count) => Edge(rtwt_id, orig_id, count) } )
												 
		val twitterGraph: Graph[(String, Long, String), Int] = Graph.apply(userRecords, userConnectionEdges)
				
		// Group the number of retweets by location and print out the top 5 locations
		println("Chase gets the most retweet impressions from the following 5 markets:")
		twitterGraph.triplets.filter( x => x.dstAttr._1 == "Chase" )
							 .map( x => (x.srcAttr._3, x.srcAttr._2 * x.attr) )							 
							 .reduceByKey(_ + _)
							 .collect.toSeq.sortBy(- _._2)
							 .take(5)
							 .foreach( x => println(x._1) )
							 
		println("Chase provides the most retweet impressions to the following 5 markets:")
		twitterGraph.triplets.filter( x => x.dstAttr._1 != "Chase" )
							 .map( x => (x.srcAttr._3, x.srcAttr._2 * x.attr) )
							 .reduceByKey(_ + _)
							 .collect.toSeq.sortBy(- _._2)
							 .take(5)
							 .foreach( x => println(x._1) )
  }
  
  def testLinearModels(trainingData : RDD[LabeledPoint], testData : RDD[LabeledPoint]) {
		// Fit training data to OLS, Ridge, and Lasso models and then print out
		// train/test MSE
		
		// Scale the features - learned from http://stackoverflow.com/questions/26259743/spark-mllib-linear-regression-model-intercept-is-always-0-0
		val scaler = new StandardScaler(withMean = true, withStd = true)
					   .fit(trainingData.map(x => x.features))
					   
		val scaledTrainingData = trainingData.map(x => 
								  LabeledPoint(x.label, 
									 scaler.transform(Vectors.dense(x.features.toArray))))		
		
		// Calibrate the models
		val linear_model = new LinearRegressionWithSGD().setIntercept(true).run(scaledTrainingData)
		val ridge_model = new RidgeRegressionWithSGD().setIntercept(true).run(scaledTrainingData)
		val lasso_model = new LassoWithSGD().setIntercept(true).run(scaledTrainingData)

		// Print train/test MSE for each model
		printLinearModelTrainTestMSE("OLS", linear_model, trainingData, testData)
		printLinearModelTrainTestMSE("Ridge", ridge_model, trainingData, testData)
		printLinearModelTrainTestMSE("Lasso", lasso_model, trainingData, testData)  
  }
  
  def printLinearModelTrainTestMSE( s : String, model : GeneralizedLinearModel, trainingData : RDD[LabeledPoint], testData : RDD[LabeledPoint]) {
		
		// Print train/test MSE for a linear model
		val trainMSE = calcLinearModelMSE( model, trainingData )
		val testMSE = calcLinearModelMSE( model, testData )
		
		println( "Model " + s + " has a training MSE of " + trainMSE + " and a test MSE of " + testMSE )
  }
  
  def calcLinearModelMSE( model : GeneralizedLinearModel, model_data : RDD[LabeledPoint] ) : Double = { 
  
		// Calculate MSE for a linear model
		val valuesAndPreds = model_data.map { point =>
				val prediction = model.predict(point.features)
				(point.label, prediction)			  
			}
			
		return valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()  
  }
  
  def testRandomForest(trainingData : RDD[LabeledPoint], testData : RDD[LabeledPoint]) {
  		// Fit training data to Random Forest model and then print out train/test MSE
		
		val categoricalFeaturesInfo = Map[Int, Int]()
		val numTrees = List(5, 10, 15, 20, 25) 
		val featureSubsetStrategy = "auto" // Let the algorithm choose.
		val impurity = "variance"
		val maxDepths = List(3, 4, 5)
		val maxBins = List(30, 40, 50, 60)

		var bestModel: Option[RandomForestModel] = None
		var bestTestMSE = Double.MaxValue

		// Test to find best collection of parameters based on lowest test MSE
		for( numTree <- numTrees; maxDepth <- maxDepths; maxBin <- maxBins ) {
			val forest_model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
			  numTree, featureSubsetStrategy, impurity, maxDepth, maxBin)
			  
			val testMSE = calcForestModelMSE(forest_model, testData)
			
			if(testMSE < bestTestMSE){
				bestModel = Some(forest_model)
				bestTestMSE = testMSE
			}			
		  }
		  
		val trainMSE = calcForestModelMSE( bestModel.get, trainingData )
		val testMSE = calcForestModelMSE( bestModel.get, testData )
		
		println( "The best Random Forest Model has a training MSE of " + trainMSE + " and a test MSE of " + testMSE )		  
		println( "The Forest parameters are: " + bestModel.get.toDebugString )
  }
  
  def calcForestModelMSE( model : RandomForestModel, model_data : RDD[LabeledPoint] ) : Double = { 
		
		// Calculate MSE for a Random Forest model
		val valuesAndPreds = model_data.map { point =>
				val prediction = model.predict(point.features)
				(point.label, prediction)			  
			}
			
		return valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()  
  }  
}