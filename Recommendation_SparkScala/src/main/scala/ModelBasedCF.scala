import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import java.util.Date
import java.io._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs._
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object ModelBasedCF {

  def main(args: Array[String]):Unit = {
    var start_time = new Date().getTime
    val conf = new SparkConf().setAppName("response_count").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.getOrCreate

    var train_raw = sc.textFile(args(0))
    var test_raw = sc.textFile(args(1))
    val train_header = train_raw.first()
    train_raw = train_raw.filter(row => row != train_header)
    val test_header = test_raw.first()
    test_raw = test_raw.filter(row => row != test_header)
    val n = test_raw.count()

    //convert user to user_id
    var train_rating = train_raw.map(_.split(','))
    var test_rating = test_raw.map(_.split(','))
    var all_rating = train_rating++test_rating
    var user = all_rating.map(x=>(x(0),1)).reduceByKey((a, b) => a + b)
    val user_encode = user.map(x=>x._1).zipWithUniqueId().collect().toMap
    val encode_user = user_encode.toList.map(x=>(x._2,x._1)).toMap
    var train_rating_1 = train_rating.map(x=>(user_encode(x(0)).toInt,x(1),x(2)))
    var test_rating_1 = test_rating.map(x=>(user_encode(x(0)).toInt,x(1),x(2)))

    //conver product to product_id
    var product = all_rating.map(x=>(x(1),1)).reduceByKey((a, b) => a + b)
    val product_encode = product.map(x=>x._1).zipWithUniqueId().collect().toMap
    var encode_product = product_encode.toList.map(x=>(x._2,x._1)).toMap
    var train_rating_2 = train_rating_1.map(x=>(x._1,product_encode(x._2).toInt,x._3.toDouble))
    val length = train_rating_2.count

    var test_rating_2 = test_rating_1.map(x=>(x._1,product_encode(x._2).toInt,x._3.toDouble))

    //ALS
    val ratings = train_rating_2.map(_ match { case (user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)})
    val test_ratings = test_rating_2.map(_ match { case (user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)})
    val rank = 20
    val numIterations = 20
    val model = ALS.train(ratings, rank, numIterations, 0.3)

    // Evaluate the model on rating data
    val test_Products = test_ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }
    var predictions =
      model.predict(test_Products).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }.map(x => (x._1, x._2.toString.toDouble))
//    val pre_map = predictions.collect().toMap

    var predict_1_ui = predictions.collectAsMap()
    var rest = test_rating_2.map(x => ((x._1, x._2)))
    var rest_train = train_rating_2.map(x => ((x._1, x._2)))

    var item_avg = train_rating_2.groupBy(x => (x._2)).map(record => {
      val rate = record._2.map(x => (x._3))
      val avg = rate.sum.toDouble / rate.size.toDouble
      (record._1, avg)
    }).collect().toList.toMap
    var rest_rate = rest.map(x => {
      if (item_avg.contains(x._2)) {(x, item_avg(x._2))}
      else {(x, 4.0)}
    })
    var test_true = test_rating_2.map(x => ((x._1, x._2), x._3))
    var predict = rest_rate.leftOuterJoin(predictions).map(x=>{
      var rate_pair = x._2
      var rate1 = rate_pair._1
      var rate2 = rate_pair._2.toList
      var rate = rate1
      if (rate2.size > 0){rate = (rate1 + rate2(0))/2}
      (x._1, rate)
    })
    var ratesAndPreds = predict.join(test_true)
    val SE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.sum()
    val RMSE = math.sqrt(SE/n)

    //rate_level
    var error = ratesAndPreds.map { case((user, product), (r1, r2)) =>
      math.abs(r1-r2)}
      .map(err=>{
      if (err.toFloat < 1 && err.toFloat >= 0){
        0
      }else if (err.toFloat >= 1 && err.toFloat < 2) {
        1
      }else if (err.toFloat >= 2 && err.toFloat < 3) {
        2
      }else if (err.toFloat >= 3 && err.toFloat < 4) {
        3
      }else if (err.toFloat >= 4) {
        4
      }
    })

//    error.foreach(println)
    var level_err = error.map(x=>(x,1)).reduceByKey((a, b) => a + b)
//    level_err.foreach(println)

    var level1 = level_err.filter(_._1==0).first()._2
    var level2 = level_err.filter(_._1==1).first()._2
    var level3 = level_err.filter(_._1==2).first()._2
    var level4 = level_err.filter(_._1==3).first()._2
    var level5 = level_err.filter(_._1==4).first()._2

    println(">=0 and <1:"+ level1)
    println(">=1 and <2:"+ level2)
    println(">=2 and <3:"+ level3)
    println(">=3 and <4:"+ level4)
    println(">=4:"+ level5)
    println(s"RMSE = $RMSE")

    var time = (new Date().getTime - start_time) / 1000
    println(s"Time: $time"+"sec")

    var predictions_origin = predict.map(x=>((encode_user(x._1._1),encode_product(x._1._2)),x._2)).sortByKey().map(x=>(x._1._1,x._1._2,x._2))

    def merge(srcPath: String, dstPath: String): Unit =  {
      val hadoopConfig = new Configuration()
      val hdfs = FileSystem.get(hadoopConfig)
      FileUtil.copyMerge(hdfs, new Path(srcPath), hdfs, new Path(dstPath), false, hadoopConfig, null)
    }

    val file = "./primaryTypes.csv"
    FileUtil.fullyDelete(new File(file))
    val destinationFile= "./Xijia_Chen_ModelBasedCF.txt"
    FileUtil.fullyDelete(new File(destinationFile))

    val pre_file = predictions_origin.map(x=>{
      val line = x._1.toString+","+x._2.toString+","+x._3.toString
      line
    })

    pre_file.saveAsTextFile(file)
    merge(file, destinationFile)
    FileUtil.fullyDelete(new File(file))


  }

}
