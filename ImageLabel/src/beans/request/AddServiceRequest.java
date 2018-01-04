package beans.request;

/**
 * Created by Administrator on 2016/1/27.
 */
public class AddServiceRequest {
    String service_name;
    String service_level;
    String service_parent;

    public String getService_name() {
        return service_name;
    }

    public void setService_name(String service_name) {
        this.service_name = service_name;
    }

    public String getService_level() {
        return service_level;
    }

    public void setService_level(String service_level) {
        this.service_level = service_level;
    }

    public String getService_parent() {
        return service_parent;
    }

    public void setService_parent(String service_parent) {
        this.service_parent = service_parent;
    }
}
