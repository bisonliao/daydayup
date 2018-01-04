package beans.service;

import beans.request.DeleteImageRequest;
import beans.request.SetWorkerIDRequest;
import ngse.org.JsonRPCHandler;
import ngse.org.JsonRPCResponseBase;

import javax.servlet.http.Cookie;

/**
 * Created by bisonliao on 2017/12/29.
 */
public class SetWorkerID  extends JsonRPCHandler {
    public JsonRPCResponseBase exec(SetWorkerIDRequest request)
    {


        JsonRPCResponseBase rsp = new JsonRPCResponseBase();
        String workerID = request.getWorkerID();
        if (workerID == null || workerID.length() < 3)
        {
            rsp.setStatus(1);
            rsp.setMessage("invalid worker ID");
            return rsp;
        }

        image.label.Worker worker = new image.label.Worker(workerID);
        getHttpRequest().getSession().setAttribute(workerID, worker);
        getHttpResponse().addCookie(new Cookie("workerID", workerID));


        rsp.setStatus(0);
        rsp.setMessage("123abc");
        return rsp;
    }
}
