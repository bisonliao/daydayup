package beans.request;

/**
 * Created by Administrator on 2016/1/25.
 */
public class QueryStaffListRequest {
    String staff_name;
    String staff_phone;

    public String getStaff_name() {
        return staff_name;
    }

    public void setStaff_name(String staff_name) {
        this.staff_name = staff_name;
    }

    public String getStaff_phone() {
        return staff_phone;
    }

    public void setStaff_phone(String staff_phone) {
        this.staff_phone = staff_phone;
    }
}
