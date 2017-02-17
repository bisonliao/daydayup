this example shows how to sort huge data with hadoop.
The method is classical bucket sort algorithm.
It has to use customized partitioner，just like lots of sorted buckets
Each data falls into one bucket according its value,then we sort the data within each bucket.