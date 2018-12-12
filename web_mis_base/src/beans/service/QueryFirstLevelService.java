package beans.service;

import beans.dbaccess.FirstLevelService;

import beans.request.QueryFirstLevelServiceRequest;
import beans.response.QueryFirstLevelServiceResponse;

import ngse.org.DBUtil;
import ngse.org.JsonRPCHandler;

import javax.servlet.http.Cookie;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Administrator on 2016/1/27.
 * 查一级标准服务
 */
public class QueryFirstLevelService extends JsonRPCHandler {


    public QueryFirstLevelServiceResponse exec(QueryFirstLevelServiceRequest request)
    {
        QueryFirstLevelServiceResponse resp = new QueryFirstLevelServiceResponse();

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
        List<FirstLevelService> serviceList ;


        String sql = "select first_level_service_name from t_first_level_service where type='standard' ";
        List<Object> params = new ArrayList<Object>();
        if (request.getService_name() != null && request.getService_name().length() > 0)
        {
            sql += " where first_level_service_name=? ";
            params.add(request.getService_name());
        }

        try {
            serviceList = util.findMoreRefResult(sql, params, FirstLevelService.class);

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



        resp.setService_list((ArrayList<FirstLevelService>) serviceList);
        resp.setMessage("success");
        resp.setStatus(0);

        return resp;
    }
}
