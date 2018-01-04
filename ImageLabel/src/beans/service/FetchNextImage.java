package beans.service;

import beans.request.FetchNextImageRequest;
import beans.response.FetchNextImageResponse;
import image.label.ImageLabelPair;
import image.label.Worker;
import javafx.util.Pair;
import ngse.org.JsonRPCHandler;
import ngse.org.JsonRPCResponseBase;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by bisonliao on 2017/12/29.
 */
public class FetchNextImage extends JsonRPCHandler {

    public FetchNextImageResponse exec(FetchNextImageRequest request)
    {
        FetchNextImageResponse resp = new FetchNextImageResponse();

        Worker worker = DeleteImage.fetchWorkerFromSession(this);
        if (worker == null)
        {
            resp.setStatus(1);
            resp.setMessage("please set worker ID at first");
            return resp;
        }
        ImageLabelPair pair = worker.getNext();
        if (pair == null)
        {
            resp.setStatus(2);
            resp.setMessage("NO more picture.");
            return resp;
        }

        resp.setImageFilename(pair.getImageFilename());
        resp.setLabel(pair.getLabel());
        resp.setLabelTextList(worker.getLabelTextList());

        return resp;
    }
}
