package beans.response;

import ngse.org.JsonRPCResponseBase;

/**
 * Created by Administrator on 2016/2/11.
 */
public class GetSaltResponse extends JsonRPCResponseBase {
    String salt;
    String challenge;

    public String getSalt() {
        return salt;
    }

    public void setSalt(String salt) {
        this.salt = salt;
    }

    public String getChallenge() {
        return challenge;
    }

    public void setChallenge(String challenge) {
        this.challenge = challenge;
    }
}
