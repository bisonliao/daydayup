package beans.response;

import beans.dbaccess.StaffInfo;
import ngse.org.JsonRPCResponseBase;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Administrator on 2016/1/25.
 */
public class QueryStaffListResponse extends JsonRPCResponseBase {
    public ArrayList<StaffInfo> staff_list;

    public ArrayList<StaffInfo> getStaff_list() {
        return staff_list;
    }

    public void setStaff_list(ArrayList<StaffInfo> staff_list) {
        this.staff_list = staff_list;
    }
}
