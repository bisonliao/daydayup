package ngse.org;

/**
 * Created by Administrator on 2016/1/23.
 * ajax应答的基础类，所有exec（）函数的返回类型都必须继承自该类
 * 该类定义了两个字段，status 和 message，是所有json应答字符串都必须有的
 * status==0表示成功，==100表示失败， ==99表示用户身份过期或者未登录
 */
public class JsonRPCResponseBase {
    public int getStatus() {
        return status;
    }

    public void setStatus(int status) {
        this.status = status;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    int status;
    String message;
}
