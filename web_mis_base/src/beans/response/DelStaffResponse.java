package beans.response;

import ngse.org.JsonRPCResponseBase;

/**
 * Created by Administrator on 2016/1/26.
 */
public class DelStaffResponse extends JsonRPCResponseBase {
    public int getDeleteNumber() {
        return deleteNumber;
    }

    public void setDeleteNumber(int deleteNumber) {
        this.deleteNumber = deleteNumber;
    }

    int deleteNumber;
}
