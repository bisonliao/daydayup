topic:  

Users of app 390854908 play videos.  Each record contains these fields:

1.  appid
2. userid
3. starttime(unixtimestamp in ms)
4. endtime
5. accumulated jittertime in this play session(ms).

We need to analyze :this app, what is the jitter rate for each day ?



1、python script used to create original data randomly:

```python
#!/usr/bin/python3
import time
import random

t = time.time()
t = int(round(t * 1000000))

for i in range(1000000):
    userid = "u%d"%i; 
    streamid = "s%d"%(i/7)
    timestart = t - random.randint(80000*1000, 80000*1000*100)
    timeend = timestart + 1000 * random.randint(30, 7200)
    jitter = round((timeend-timestart) * (random.randint(0, 100)/200.0))
    record = "390854908,%s,%s,%d,%d,%d"%( userid, streamid,  timestart, timeend, jitter)
    print(record)
```



2、upload to hdfs

```shell
 su - hadoop; hdfs dfs  -copyFromLocal ./jitter.txt   /HiveDirect/jitter_info1.txt
```



3、this app, what is the jitter rate for each day ?

3.1  by RDD:

```python
import  time
import datetime
from pyspark import SparkContext
from pyspark.sql.functions import udf
from pyspark.sql.functions import from_unixtime
from pyspark.sql.types import StringType

sc = SparkContext("local", "first app")

rdd=sc.textFile("/HiveDirect/jitter_info1.txt")
t=rdd.map(lambda x:x.split(","))

duration = t.map(lambda x: ( sec2datetime(int(x[3])/1000), int(x[4])-int(x[3]) ) )
total = duration.reduceByKey(lambda a,b:a+b)

jitter = t.map(lambda x: ( sec2datetime(int(x[3])/1000), int(x[5]) ) )
jitter = jitter.reduceByKey(lambda a,b: a+b)

rate = jitter.join(total)
rate = rate.map(lambda x:(x[0], x[1][0]*1.0/x[1][1]) )
print(rate.sortByKey().take(100))
```

3.2 by DataFrame:

```python
from pyspark import SparkContext
from pyspark.sql.functions import udf
from pyspark.sql.functions import from_unixtime


from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("bisonliao") \
    .getOrCreate()

df = spark.read.csv("/HiveDirect/jitter_info1.txt")
df=df.toDF("appid", "userid", "streamid", "starttime", "endtime", "jitter")
df=df.withColumn("dur", df.endtime-df.starttime)
df=df.withColumn("jitter", df.jitter-0)
df=df.withColumn("date", from_unixtime(df.starttime/1000, "yyyy-MM-dd"))
df=df.groupBy("date").sum()
df = df.toDF("date", "jitter_total", "dur_total")
df=df.withColumn("rate", df.jitter_total/df.dur_total)
df.show()
```

4、commit to spark：

```shell
su - hadoop; spark-submit statistic2.py
```

result:

```shell
+----------+---------------+------------+-------------------+
|      date|   jitter_total|   dur_total|               rate|
+----------+---------------+------------+-------------------+
|2021-06-17|1.0073368545E10|4.0173861E10| 0.2507443470519301|
|2021-05-20|  9.677666775E9|3.8982234E10| 0.2482583931695654|
|2021-07-18|  9.973982265E9|3.9707169E10|0.25118845075558016|
|2021-07-01|  9.807160895E9|3.9187915E10| 0.2502598286997407|
|2021-07-07|   9.80206177E9|3.9096859E10|0.25071225721738927|
|2021-04-25|  9.754113015E9|3.9128853E10|0.24928185385347226|
|2021-06-03|   9.92418932E9|3.9501428E10| 0.2512362165742464|
|2021-06-26|  9.853293565E9|3.9408885E10|0.25002720998069344|
|2021-06-23|  9.866856315E9|3.9517323E10|0.24968432995828185|
|2021-05-08|  9.701159165E9| 3.910367E10|0.24808820156778125|
|2021-06-27|  9.931203405E9|3.9631491E10| 0.2505886898123515|
|2021-04-22|    2.8052122E8|  1.135185E9|0.24711498125856138|
|2021-07-12|1.0025384725E10|3.9685583E10|0.25262032121337363|
|2021-05-13|   9.67240546E9|3.8701462E10|0.24992351606768756|
|2021-06-24|   9.91850577E9|3.9808046E10|0.24915831764261928|
|2021-07-03|  9.866887385E9|3.9270711E10|0.25125308744728353|
|2021-06-11|   9.89457616E9| 3.923638E10| 0.2521786199440417|
|2021-05-15| 1.011639308E10| 4.017138E10|0.25183085769022623|
|2021-04-26|  9.815694895E9|3.9271715E10|0.24994311796670962|
|2021-07-13|   9.78228551E9|3.9049388E10| 0.2505105972467481|
+----------+---------------+------------+-------------------+
only showing top 20 rows
```

