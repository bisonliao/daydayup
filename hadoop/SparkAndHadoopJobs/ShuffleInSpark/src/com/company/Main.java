package com.company;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.security.SecureRandom;

public class Main {
    public static SecureRandom rnd = null;
    static
    {
        try {
            rnd = java.security.SecureRandom.getInstance("SHA1PRNG");
            rnd.setSeed((long) (Math.random() * 1000000));
        }catch (Exception e)
        {
            rnd = null;
        }
    }

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setMaster("spark://namenode:7077").setAppName("CalcAvgInSpark");
        JavaSparkContext sc = new JavaSparkContext(conf);


        ///下面这个例子，是如何将已经排好序的数据随机打乱，类似洗牌
      //已经排好序的数据集
        JavaRDD<String> lines = sc.textFile("hdfs://namenode:9000/integers.txt");
        JavaRDD<Long> integers = lines.map(new Function<String, Long>() {
            @Override
            public Long call(String s) throws Exception {
                return new Long(s);
            }
        });
    //洗牌打乱
        JavaPairRDD<Long, Long> pairRdd = integers.mapToPair(new PairFunction<Long, Long, Long>() {
            @Override
            public scala.Tuple2<Long, Long> call(Long aLong) throws Exception {

                long index = rnd.nextLong();
                return new scala.Tuple2<Long, Long>(index, aLong);
            }
        });
        JavaPairRDD<Long, Long> pairRdd2 = pairRdd.sortByKey();
        JavaRDD<Long> shuffledRDD = pairRdd2.map(new Function<Tuple2<Long,Long>, Long>() {

            @Override
            public Long call(Tuple2<Long, Long> longLongTuple2) throws Exception {
                return longLongTuple2._2();
            }
        });
        shuffledRDD.saveAsTextFile("hdfs://namenode:9000/NotSorted");


    }
}
