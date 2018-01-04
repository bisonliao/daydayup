package beans.service;

import beans.dbaccess.StaffInfo;
import beans.request.QueryStaffListRequest;
import beans.response.QueryStaffListResponse;
import ngse.org.DBUtil;
import ngse.org.JsonRPCHandler;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Administrator on 2016/1/25.
 * 查询用户列表
 */
public class QueryStaffList extends JsonRPCHandler {

    public QueryStaffListResponse exec(QueryStaffListRequest request)
    {
        QueryStaffListResponse resp = new QueryStaffListResponse();
        String result = checkIdentity();
        if (!result.equals("success"))
        {
            resp.setStatus(99);
            resp.setMessage(result);
            return resp;
        }
        DBUtil util = new DBUtil();
        if (util.getConnection() == null)
        {
            resp.setStatus(100);
            resp.setMessage("db connect failed!");
            return resp;
        }
        List<StaffInfo> staffInfoList ;
        //System.out.printf("name:%s, phone:%s\n", request.getStaff_name(), request.getStaff_phone());

        String sql = "select staff_name, staff_phone from t_staff ";
        List<Object> params = new ArrayList<Object>();
        if (request.getStaff_name() != null && request.getStaff_name().length() > 0)
        {
            sql += " where staff_name=? ";
            params.add(request.getStaff_name());
        }
        else if (request.getStaff_phone() != null && request.getStaff_phone().length() > 0)
        {
            sql += " where staff_phone=? ";
            params.add(request.getStaff_phone());
        }
        try {
            staffInfoList = util.findMoreRefResult(sql, params, StaffInfo.class);

        }
        catch (Exception e)
        {
            resp.setStatus(100);
            resp.setMessage("db query exception!");
            e.printStackTrace();
            return resp;
        }
        finally {
            util.releaseConn();
        }



        resp.setStaff_list((ArrayList<StaffInfo>)staffInfoList);
        resp.setMessage("success");
        resp.setStatus(0);

        return resp;


    }
}
