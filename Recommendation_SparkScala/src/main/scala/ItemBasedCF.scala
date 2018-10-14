import java.io.File
import java.util.Date
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

object ItemBasedCF {

  def main(args: Array[String]):Unit = {
    var start_time = new Date().getTime
    val conf = new SparkConf().set("spark.executor.memory", "1g").set("spark.driver.memory", "4g").setAppName("response_count").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.getOrCreate

    //convert raw
    var train_raw = sc.textFile(args(0))
    var test_raw = sc.textFile(args(1))
    val train_header = train_raw.first()
    train_raw = train_raw.filter(row => row != train_header)
    val test_header = test_raw.first()
    test_raw = test_raw.filter(row => row != test_header)
    val n = test_raw.count()
    var train_rating = train_raw.map(_.split(','))
    var test_rating = test_raw.map(_.split(','))
    var all_rating = train_rating.union(test_rating)
    var user = all_rating.map(x => (x(0), 1)).reduceByKey((a, b) => a + b)
    val user_encode = user.map(x => x._1).zipWithUniqueId().collect().toMap
    val encode_user = user_encode.toList.map(x => (x._2, x._1)).toMap
    var train_rating_1 = train_rating.map(x => (user_encode(x(0)).toInt, x(1), x(2)))
    var test_rating_1 = test_rating.map(x => (user_encode(x(0)).toInt, x(1), x(2)))
    var product = all_rating.map(x => (x(1), 1)).reduceByKey((a, b) => a + b)
    val product_encode = product.map(x => x._1).zipWithUniqueId().collect().toMap
    var encode_product = product_encode.toList.map(x => (x._2, x._1)).toMap
    var train_rating_2 = train_rating_1.map(x => (x._1, product_encode(x._2).toInt, x._3.toDouble))
    var test_rating_2 = test_rating_1.map(x => (x._1, product_encode(x._2).toInt, x._3.toDouble))
    var all_rating_encode = train_rating_2 ++ test_rating_2
    var train_rating_3 = train_rating_2.groupBy(x => x._2).filter(x => x._2.size > 25).flatMap(x => x._2)
    var train_rating_3_map = train_rating_2.map(x => ((x._1, x._2), x._3)).collect().toList.toMap
    var user_key = train_rating_3.map(x => (x._1, x))
    var user_join = user_key.join(user_key).filter(x => x._2._1._2 <= x._2._2._2).map(x => (x._1, ((x._2._1._2, x._2._1._3), (x._2._2._2, x._2._2._3))))
    var item_pair = user_join.map(x => ((x._2._1._1, x._2._2._1), (x._1, x._2._1._2, x._2._2._2))) //  ((i1,i2),(u,r1,r2))

    //compute normalized vectors
    var item_pair_sum = item_pair.map(x => (x._1, (x._2._2, x._2._3))).reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))// ((i1,i2), (r1.sum, r2.sum))
    var item_pair_count = item_pair.map(x => (x._1, 1)).reduceByKey((x, y) => (x + y))// ((i1,i2), common user_count)
    var item_pair_avg = item_pair_sum.join(item_pair_count) // ((i1,i2),((r1.sum,r2.sum),user_count))
      .map(x => (x._1, (x._2._1._1 / x._2._2, x._2._1._2 / x._2._2))).collect().toList.toMap// ((i1,i2),(r1.avg, r2.avg))
    var item_pair_normalize = item_pair.map(record => {
      val itempair = record._1
      val rate = record._2
      val norm_rate = (rate._1, rate._2 - item_pair_avg(itempair)._1, rate._3 - item_pair_avg(itempair)._2)
      (itempair, norm_rate)
    }) // ((i1,i2),(u,r1_norm,r2_norm))

    //compute half similiarity matrix
    var item_pair_simi = item_pair_normalize.groupByKey().map(record => {
      val key = record._1
      val value = record._2
      val dot = value.map(x => x._2 * x._3).sum
      val r1_dist = math.sqrt(value.map(x => x._2 * x._2).sum)
      val r2_dist = math.sqrt(value.map(x => x._3 * x._3).sum)
      if (dot == 0 || r1_dist == 0 || r2_dist == 0) {
        (key, 0)
      }
      else {
        (key, dot / (r1_dist * r2_dist))
      }
    }) //((item1,item2),w)

    //build up whole similarity matrix
    var item_pair_simi2 = item_pair_simi.map(x => ((x._1._2, x._1._1), x._2))
    var similarity = (item_pair_simi ++ item_pair_simi2)
    var simi_map = similarity.collectAsMap()
    print("************" + similarity.count())

    // find candidate items from train set
    var test_user = test_rating_2.map(x => (x._1, 1)).groupByKey().map(x => (x._1, 1))
    var train_user_item = train_rating_3.map(x => (x._1, x._2))
    var test_user_trainitem = test_user.join(train_user_item).map(x => (x._1, x._2._2))
    var test_user_group = test_user_trainitem.groupByKey()
    var test_user_item = test_rating_2.map(x => (x._1, x._2))
    var test_user_all = test_user_group.join(test_user_item).map(x => ((x._1, x._2._2), x._2._1)) // ((test_user,test_item),(i1_train,i2_train...))
    var trainitem_w = test_user_all.map(record => {
      var test_user_item = record._1
      val train_item = record._2
      val trainitem_w = train_item.map(x => (test_user_item._2, x)).map(x => {
        if (simi_map.contains(x)) {(x._2, simi_map(x))}
        else {(x._2, -10.0)}
      })
      (test_user_item, trainitem_w)
    }) // ((test_user, test_item),((i1_train,w),(i2_train,w),.....))

    var trainitem_rate = test_user_all.map(record => {
      val test_user_item = record._1
      val train_item = record._2
      var trainuser_trainitem = train_item.map(x => (test_user_item._1, x))
      val trainitem_rate = trainuser_trainitem.map(x => {
        if (train_rating_3_map.contains(x)) {(x._2, train_rating_3_map(x))}
        else {(x._2, -10.0)}})
      (test_user_item, trainitem_rate)
    }) // ((test_user, test_item),((i1_train, rate),...))

    var predict_1 = trainitem_rate.join(trainitem_w).map(record => {
      val test_user_item = record._1
      val trainitem_w = record._2._2
      val trainitem_rate = record._2._1.toList.toMap
      var trainitem_w_rate_ = trainitem_w.map(x => {
        if (trainitem_rate.contains(x._1)) {(x._1, x._2, trainitem_rate(x._1))}
        else {(x._1, x._2, -1)}
      }).filter(x => (x._2.toString().toDouble != -10.0 & x._2.toString().toDouble > 0 & x._3.toString().toDouble != -10.0 & x._3.toString().toDouble != -1.0))
      var trainitem_w_rate = trainitem_w_rate_.toList.sortBy(x => (x._2.toString().toDouble)).take(5)
      val rate_w = trainitem_w_rate.map(x => (x._2.toString.toDouble * x._3.toString.toDouble)).sum
      val w = trainitem_w_rate.map(x => math.abs(x._2.toString.toDouble)).sum
      if (rate_w == 0 || w == 0) {(test_user_item, 0)}
      else {(test_user_item, rate_w / w)}
    }).filter(x => (x._2 != 0)).map(x => (x._1, x._2))
    var test_true = test_rating_2.map(x => ((x._1, x._2), x._3))
    var rest = test_rating_2.map(x => ((x._1, x._2)))
    var item_avg = train_rating_2.groupBy(x => (x._2)).map(record => {
      val rate = record._2.map(x => (x._3))
      val avg = rate.sum.toDouble / rate.size.toDouble
      (record._1, avg)
    }).collect().toList.toMap
    var rest_rate = rest.map(x => {
      if (item_avg.contains(x._2)) {(x, item_avg(x._2))}
      else {(x, 4.0)}
    })
    var predict = rest_rate.leftOuterJoin(predict_1).map(x=>{
      var rate_pair = x._2
      var rate1 = rate_pair._1
      var rate = rate1
      if (rate_pair._2 != None){rate = (rate1 + rate_pair._2.toList(0).toString.toDouble)/2}
      (x._1, rate)
    })
    var ratesAndPreds = predict.join(test_true)
    var length = ratesAndPreds.collect().size

    val SE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1.toString().toDouble - r2.toDouble)
      err * err
    }.sum()
    val RMSE = math.sqrt(SE / n)

    //rate_level
    var error = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      math.abs(r1.toString().toDouble - r2)
    }
      .map(err => {
        if (err.toFloat < 1 && err.toFloat >= 0) {
          0
        } else if (err.toFloat >= 1 && err.toFloat < 2) {
          1
        } else if (err.toFloat >= 2 && err.toFloat < 3) {
          2
        } else if (err.toFloat >= 3 && err.toFloat < 4) {
          3
        } else if (err.toFloat >= 4) {
          4
        }
      })

    var level_err = error.map(x => (x, 1)).reduceByKey((a, b) => a + b)
    var level1 = level_err.filter(_._1 == 0).first()._2
    var level2 = level_err.filter(_._1 == 1).first()._2
    var level3 = level_err.filter(_._1 == 2).first()._2
    var level4 = level_err.filter(_._1 == 3).first()._2
    var level5 = level_err.filter(_._1 == 4).first()._2

    println(">=0 and <1:" + level1)
    println(">=1 and <2:" + level2)
    println(">=2 and <3:" + level3)
    println(">=3 and <4:" + level4)
    println(">=4:" + level5)
    println(s"RMSE = $RMSE")

    var time = (new Date().getTime - start_time) / 1000
    println(s"Time: $time"+"sec")

    var predictions_origin = predict.map(x => ((encode_user(x._1._1), encode_product(x._1._2)), x._2)).sortByKey().map(x => (x._1._1, x._1._2, x._2))

    def merge(srcPath: String, dstPath: String): Unit = {
      val hadoopConfig = new Configuration()
      val hdfs = FileSystem.get(hadoopConfig)
      FileUtil.copyMerge(hdfs, new Path(srcPath), hdfs, new Path(dstPath), false, hadoopConfig, null)
    }

    val file = "./primaryTypes"
    FileUtil.fullyDelete(new File(file))
    val destinationFile = "./Xijia_Chen_ItemBasedCF.txt"
    FileUtil.fullyDelete(new File(destinationFile))

    val pre_file = predictions_origin.map(x => {
      val line = x._1.toString + "," + x._2.toString + "," + x._3.toString
      line
    })

    pre_file.saveAsTextFile(file)
    merge(file, destinationFile)
    FileUtil.fullyDelete(new File(file))
  }
}
