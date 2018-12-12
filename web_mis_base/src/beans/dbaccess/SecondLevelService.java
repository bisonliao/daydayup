package beans.dbaccess;

import ngse.org.ServletConfig;

import java.io.File;

/**
 * Created by Administrator on 2016/1/25.
 */
public class SecondLevelService {
    String second_level_service_name;
    String first_level_service_name;


    public String getSecond_level_service_name() {
        return second_level_service_name;
    }

    public void setSecond_level_service_name(String second_level_service_name) {
        this.second_level_service_name = second_level_service_name;
    }

    public String getFirst_level_service_name() {
        return first_level_service_name;
    }

    public void setFirst_level_service_name(String first_level_service_name) {
        this.first_level_service_name = first_level_service_name;
    }
}
