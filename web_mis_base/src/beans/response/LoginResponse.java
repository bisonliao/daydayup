package beans.response;

import ngse.org.JsonRPCResponseBase;

/**
 * Created by Administrator on 2016/2/10.
 */
public class LoginResponse extends JsonRPCResponseBase {
    String staff_name;
    String ticket;

    public String getStaff_name() {
        return staff_name;
    }

    public void setStaff_name(String staff_name) {
        this.staff_name = staff_name;
    }

    public String getTicket() {
        return ticket;
    }

    public void setTicket(String ticket) {
        this.ticket = ticket;
    }
}
