这个例子是：输入 <uin 好友数>这样的一行一行的记录，统计出 具有某个好友数的具体的用户数量，即输出 <好友个数  具有该好有个数的用户数量>

使用到两次map。


第一次map，避免具有某个好友数的用户量特别多，超出单机所能处理大小，所以输出的key是好友个数-随机下标，随机下标范围[0, 1000)：
输入：uin 好友个数 ，好友个数的范围[0, 1000]
输出：key 1，其中key是 好友个数-随机下标，随机下标范围[0, 1000)，作用是避免某一个“好友个数”的uin太多导致reduce过程中单机处理不过来


第二次map
输入：好友个数-随机下标 数字1的列表
输出：好友个数 uin个数

reduce
输入：好友个数 uin个数
输出：好友个数 对应的uin个数