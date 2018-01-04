package beans.service;

import beans.request.ConfirmImageRequest;
import beans.request.SaveWorkRequest;
import image.label.Worker;
import ngse.org.JsonRPCHandler;
import ngse.org.JsonRPCResponseBase;

/**
 * Created by bisonliao on 2017/12/28.
 */
public class SaveWork extends JsonRPCHandler {



    public JsonRPCResponseBase exec(SaveWorkRequest request)
    {
        JsonRPCResponseBase rsp = new JsonRPCResponseBase();

        Worker worker = DeleteImage.fetchWorkerFromSession(this);
        if (worker == null)
        {
            rsp.setStatus(1);
            rsp.setMessage("please set worker ID at first");
            return rsp;
        }
        try {
            worker.dump();
        }
        catch (Exception e)
        {
            rsp.setStatus(2);
            rsp.setMessage(e.getMessage());
            return rsp;
        }

        rsp.setStatus(0);
        rsp.setMessage("ok");
        return rsp;
    }
}
