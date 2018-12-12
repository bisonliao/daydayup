package beans.response;

import beans.dbaccess.FirstLevelService;
import ngse.org.JsonRPCResponseBase;

import java.util.ArrayList;

/**
 * Created by Administrator on 2016/1/27.
 */
public class QueryFirstLevelServiceResponse extends JsonRPCResponseBase {
    ArrayList<FirstLevelService> service_list;

    public ArrayList<FirstLevelService> getService_list() {
        return service_list;
    }

    public void setService_list(ArrayList<FirstLevelService> service_list) {
        this.service_list = service_list;
    }
}
