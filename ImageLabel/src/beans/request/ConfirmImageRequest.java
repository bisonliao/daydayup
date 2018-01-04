package beans.request;

/**
 * Created by bisonliao on 2017/12/28.
 */
public class ConfirmImageRequest {
    String imageFilename;
    int label;

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public String getImageFilename() {
        return imageFilename;
    }

    public void setImageFilename(String imageFilename) {
        this.imageFilename = imageFilename;
    }
}
