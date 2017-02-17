import org.apache.spark.Partitioner;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.rdd.ShuffledRDD;
import scala.Tuple2;

import java.io.Serializable;
import java.security.SecureRandom;
import java.security.Security;
import java.util.List;

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



        /*
         * ������������ǣ����� <uin ������>������һ��һ�еļ�¼��ͳ�Ƴ� ����ĳ���������ľ�����û������������ <���Ѹ���  ���иú��и������û�����>
         */

        JavaRDD<String> lines = sc.textFile("hdfs://namenode:9000/frdNum.txt");

        //����ԭʼ���ݣ������� <������-suffix 1>�����ļ�¼�� ����ֱ�۵���⣺����<������ 1>�����ļ�¼Ȼ��reduce�ȿ��Եõ����
        // ���Ǿ���ĳ�������������û����ܷǳ��࣬����mapreduce��map�׶β����ĵ���key value��¼�����������������Զ�key���н�һ���ķֲ�
        // reduce���ٽ�һ��map reduce
        JavaPairRDD<String, Long> pairRdd = lines.mapToPair(new PairFunction<String, String, Long>() {
            @Override
            public Tuple2<String, Long> call(String line) throws Exception {
                int sepIndex = line.indexOf(" ");
                if (sepIndex <= 0) {
                    return null;
                }
                String uinStr = line.substring(0, sepIndex);
                String frdNumStr = line.substring(sepIndex + 1);
                int rnd = (int) (Math.random() * 1000);
                if (rnd < 0) {
                    rnd = -rnd;
                }
                String keyStr = frdNumStr + "-" + rnd;
                return new Tuple2<String, Long>(keyStr, new Long(1));
            }
        });
        pairRdd = pairRdd.reduceByKey(new Function2<Long, Long, Long>() {
            @Override
            public Long call(Long aLong, Long aLong2) throws Exception {
                return new Long(aLong.longValue() + aLong2.longValue());
            }
        });
        JavaPairRDD<Long, Long> pairRDD2 = pairRdd.mapToPair(new PairFunction<Tuple2<String, Long>, Long, Long>() {
            @Override
            public Tuple2<Long, Long> call(Tuple2<String, Long> stringLongTuple2) throws Exception {
                String keyStr = stringLongTuple2._1();
                int sepIndex = keyStr.indexOf("-");
                if (sepIndex <= 0) {
                    return null;
                }
                String frdNumStr = keyStr.substring(0, sepIndex);
                return new Tuple2<Long, Long>(new Long(frdNumStr), stringLongTuple2._2());

            }
        });
        pairRDD2 = pairRDD2.reduceByKey(new Function2<Long, Long, Long>() {
            @Override
            public Long call(Long aLong, Long aLong2) throws Exception {
                return new Long(aLong.longValue() + aLong2.longValue());
            }
        });
        pairRDD2.saveAsTextFile("hdfs://namenode:9000/output");













    }
}
