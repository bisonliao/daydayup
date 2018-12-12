package beans.service;

import beans.dbaccess.DBAnalyzeInfo;
import beans.request.DelServiceRequest;
import beans.response.DelServiceResponse;
import ngse.org.DBUtil;
import ngse.org.JsonRPCHandler;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import org.apache.log4j.Logger;

/**
 * Created by Administrator on 2016/1/27.
 * 删除标准服务
 */
public class DelService extends JsonRPCHandler{

    private boolean checkIfHasSecondLevel(String first_level_service_name, DBUtil util)
    {
       String sql = "select count(*) as record_number from t_second_level_service where first_level_service_name=?";
        List<Object> params = new ArrayList<Object>();
        params.add(first_level_service_name);
        try
        {
            DBAnalyzeInfo dbinfo = util.findSimpleRefResult(sql, params, DBAnalyzeInfo.class);
            if (dbinfo.getRecord_number() > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return true;
        }
    }


    public DelServiceResponse exec(DelServiceRequest request)
    {
        Logger logger = Logger.getLogger(this.getClass().getName());
    DelServiceResponse response = new DelServiceResponse();


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
        response.setMessage("The name/level of service to be deled should NOT be empty.");
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
    if (request.getService_level().equals("first_level")) {
        //检查一下一级服务下还有没有二级服务，有的话不能删除
        if (checkIfHasSecondLevel(request.getService_name(), util) != false)
        {
            response.setMessage("还有二级服务挂靠在该一级服务下，不能删除该服务.");
            response.setStatus(100);
            return response;
        }

        sql = "delete from t_first_level_service where first_level_service_name=?";
        logger.info(sql);
        params.add(request.getService_name());

    }
    else
    {
        sql = "delete from t_second_level_service where second_level_service_name=? and first_level_service_name=?";
        logger.info(sql);
        params.add(request.getService_name());
        params.add(request.getService_parent());
    }


    try {
        int delNum = util.updateByPreparedStatement(sql, params);
        if (delNum == 1)
        {
            //相关的一些信息也应该删除掉，例如IP、config、IDL等等
            if (request.getService_level().equals("second_level")) {

            }
            response.setDelNum(delNum);
            response.setMessage("success");
            response.setStatus(0);
            return response;
        }
        else
        {
            response.setDelNum(delNum);
            response.setMessage("delete record number is "+delNum);
            response.setStatus(100);
            return response;
        }

    }
    catch (SQLException e)
    {
        response.setMessage("del record failed:"+e.toString());
        response.setStatus(100);
        e.printStackTrace();
        return response;
    }
    finally {
        util.releaseConn();
    }

}
}
