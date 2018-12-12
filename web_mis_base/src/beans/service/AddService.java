package beans.service;

import beans.request.AddServiceRequest;
import beans.response.AddNewStaffResponse;
import beans.response.AddServiceResponse;
import ngse.org.DBUtil;
import ngse.org.JsonRPCHandler;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Administrator on 2016/1/27.
 * 新增标准服务，可能是一级服务，也可能是二级
 */
public class AddService extends JsonRPCHandler {
    public AddServiceResponse exec(AddServiceRequest request)
    {
        AddServiceResponse response = new AddServiceResponse();

        response.setMessage("unkown error.");
        response.setStatus(100);
        String result = checkIdentity();
        if (!result.equals("success"))
        {
            response.setStatus(99);
            response.setMessage(result);
            return response;
        }
        if (request.getService_name() == null ||
                request.getService_name().equals("") ||
                request.getService_level() == null ||
                request.getService_level().equals(""))
        {
            response.setMessage("The name/level of service to be added should NOT be empty.");
            response.setStatus(100);
            return response;
        }
        if ( request.getService_level().equals("second_level") &&
                ( request.getService_parent() == null || request.getService_parent().equals("")))
        {
            response.setMessage("The first level service name  should NOT be empty.");
            response.setStatus(100);
            return response;
        }
        DBUtil util = new DBUtil();
        if (util.getConnection() == null)
        {
            response.setMessage("DB connect failed.");
            response.setStatus(100);
            return response;
        }
        String sql;
        List<Object> params = new ArrayList<Object>();
        if (request.getService_level().equals("first_level"))
        {
             sql = "insert into t_first_level_service( first_level_service_name, type) values(?, 'standard')";

            params.add(request.getService_name());

        }
        else
        {
             sql = "insert into t_second_level_service(second_level_service_name, first_level_service_name, type) values(?,?,'standard')";

            params.add(request.getService_name());
            params.add(request.getService_parent());
        }


        try {
            int addNum = util.updateByPreparedStatement(sql, params);
            if (addNum >= 0)
            {
                response.setAddNum(addNum);
                response.setMessage("success");
                response.setStatus(0);
                return response;
            }
        }
        catch (SQLException e)
        {
            response.setMessage("add record failed:"+e.toString());
            response.setStatus(100);
            e.printStackTrace();
            return response;
        }
        finally {
            util.releaseConn();
        }
        return  response;
    }
}
