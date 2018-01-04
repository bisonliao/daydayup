package beans.service;

import beans.request.DeleteImageRequest;
import beans.request.LoginRequest;
import image.label.Worker;
import ngse.org.JsonRPCHandler;
import ngse.org.JsonRPCResponseBase;

import javax.servlet.http.Cookie;

/**
 * Created by bisonliao on 2017/12/28.
 */
public class DeleteImage  extends JsonRPCHandler {

    //可复用的代码，收拢到这里
    public static image.label.Worker fetchWorkerFromSession(JsonRPCHandler handler)
    {
        Cookie[] cookies = handler.getHttpRequest().getCookies();
        String workerID = null;
        for (int i = 0; i < cookies.length; ++i)
        {
            if (cookies[i].getName().equals("workerID"))
            {
                workerID = cookies[i].getValue();
                break;
            }
        }
        Object o = handler.getHttpRequest().getSession().getAttribute(workerID);
        if (o == null) { return null;}
        return (image.label.Worker)o;
    }



    public JsonRPCResponseBase exec(DeleteImageRequest request)
    {
        JsonRPCResponseBase rsp = new JsonRPCResponseBase();

        Worker worker = fetchWorkerFromSession(this);
        if (worker == null)
        {
            rsp.setStatus(1);
            rsp.setMessage("please set worker ID at first");
            return rsp;
        }
        worker.deleteImage(request.getImage_filename());

        rsp.setStatus(0);
        rsp.setMessage("123abc");
        return rsp;
    }
}
