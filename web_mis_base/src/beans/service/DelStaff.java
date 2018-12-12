package beans.service;

import beans.request.DelStaffRequest;
import beans.response.DelStaffResponse;
import ngse.org.DBUtil;
import ngse.org.JsonRPCHandler;
import ngse.org.JsonRPCResponseBase;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Administrator on 2016/1/26.
 * 删除用户
 */
public class DelStaff extends JsonRPCHandler {

    public DelStaffResponse exec(DelStaffRequest request)
    {
        DelStaffResponse response = new DelStaffResponse();
        response.setMessage("unkown error.");
        response.setStatus(100);
        String result = checkIdentity();
        if (!result.equals("success"))
        {
            response.setStatus(99);
            response.setMessage(result);
            return response;
        }
        if (request.getStaff_name() == null ||
                request.getStaff_name().equals(""))
        {
            response.setMessage("The name of staff to be deleted should NOT be empty.");
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
        String sql = "delete from t_staff where staff_name=?";
        List<Object> params = new ArrayList<Object>();
        params.add(request.getStaff_name());
        try {
            int delNum = util.updateByPreparedStatement(sql, params);
            if (delNum >= 0)
            {
                response.setMessage("success");
                response.setDeleteNumber(delNum);
                response.setStatus(0);
                return response;
            }
        }
        catch (SQLException e)
        {
            response.setMessage("Delete record failed:"+e.toString());
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
