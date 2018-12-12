package beans.response;

import ngse.org.JsonRPCResponseBase;

/**
 * Created by Administrator on 2016/1/27.
 */
public class DelServiceResponse extends JsonRPCResponseBase {
    int delNum;

    public int getDelNum() {
        return delNum;
    }

    public void setDelNum(int addNum) {
        this.delNum = delNum;
    }
}
