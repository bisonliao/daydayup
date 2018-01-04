package beans.service;

import beans.request.ConfirmImageRequest;
import beans.request.DeleteImageRequest;
import image.label.Worker;
import ngse.org.JsonRPCHandler;
import ngse.org.JsonRPCResponseBase;

import javax.servlet.http.Cookie;

/**
 * Created by bisonliao on 2017/12/28.
 */
public class ConfirmImage extends JsonRPCHandler {



    public JsonRPCResponseBase exec(ConfirmImageRequest request)
    {
        JsonRPCResponseBase rsp = new JsonRPCResponseBase();

        Worker worker = DeleteImage.fetchWorkerFromSession(this);
        if (worker == null)
        {
            rsp.setStatus(1);
            rsp.setMessage("please set worker ID at first");
            return rsp;
        }
        worker.confirmLabel(request.getImageFilename(), request.getLabel());

        rsp.setStatus(0);
        rsp.setMessage("123abc");
        return rsp;
    }
}
