package beans.response;

import ngse.org.JsonRPCResponseBase;

/**
 * Created by Administrator on 2016/1/27.
 */
public class AddServiceResponse extends JsonRPCResponseBase {
    int addNum;

    public int getAddNum() {
        return addNum;
    }

    public void setAddNum(int addNum) {
        this.addNum = addNum;
    }
}
