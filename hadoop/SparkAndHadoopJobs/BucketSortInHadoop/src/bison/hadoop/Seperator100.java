package bison.hadoop;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Seperator100 {

    static public Comparator<String> comp = new Comparator<String>() {
        @Override
        public int compare(String o1, String o2) {
            return o1.compareTo(o2);

        }
    };

    static public String getSeperator(String s)
    {
        int index = Collections.binarySearch(sepList, s, comp);
        /*
        the index of the search key, if it is contained in the list;

        otherwise, (-(insertion point) - 1).
        The insertion point is defined as the point at which the key would be inserted into
        the list: the index of the first element greater than the key,

        or list.size() if all elements in the list are less than the specified key. Note that this guarantees that the return value will be >= 0 if and only if the key is found.
         */
        if (index  == -1)
        {
            return sepList.get(0);
        }
        else  if (index < 0)
        {
            index += 1;
            index = -index;
            return sepList.get(index - 1);
        }
        else if (index == sepList.size())
        {
            return sepList.get(index-1);
        }
        else {
            return sepList.get(index);
        }
    }
    static public void main(String[] args)
    {
        String s = getSeperator("0mYI8TsOb4");
        System.out.println(s);

        s = getSeperator("03KNFcDUE8");
        System.out.println(s);

        s = getSeperator("0FI4uqj6vC");
        System.out.println(s);

        s = getSeperator("zpjgdDQuTQ");
        System.out.println(s);
    }
    static int getPartitionNumber()
    {
        return sepList.size();
    }
    static int getPartitionIndex(String s)
    {
        int index = Collections.binarySearch(sepList, s, comp);
        if (index < 0 || index == sepList.size())
        {
            return -1;
        }
        return index;
    }

    static private List<String> sepList = new ArrayList<String>();
    static {
        sepList.add("00a5f4096409bc29dd1496c29c5ee59c");
        sepList.add("020681a6ad3de0975441b9461e23d306");
        sepList.add("048992ab727048e7bef35a9168f22bfd");
        sepList.add("07b75abde9830f477313b0c0afa4423d");
        sepList.add("0a6db55f3cb4f06c126364c95e70c399");
        sepList.add("0aad84a2097e2da9590eaba59b1c9e26");
        sepList.add("10f54ea8a7e90094a2ed0315a5d5698d");
        sepList.add("128f1b369ad417daad56c994e11e0d06");
        sepList.add("143317650c49a289ba1102190b18822b");
        sepList.add("1514144d36f7c9f591b9b33c780183e8");
        sepList.add("166ad9744f056259d4875e50d5d4df25");
        sepList.add("17b65951485358df0873251f3867ec6a");
        sepList.add("1d06ef4ea2d5b1722025d052a42debe8");
        sepList.add("1fffdad6d8d297761e7037a3b4a6a397");
        sepList.add("2514279d3014aefe098963c43ca4df3a");
        sepList.add("2c9ee416b0681d2a0c45d8076fd2bd35");
        sepList.add("319c2750b9b60cb9b2785a35f81202a5");
        sepList.add("35f2a2db4c496fd082f587aedee8aa06");
        sepList.add("365cf0d2482c98ba83bc708bdfb501ab");
        sepList.add("36924f4174b60fac535641c7ba5e5d8c");
        sepList.add("3875a19e074ad4431f1be3449f788bc9");
        sepList.add("3c25c1f9215663ee0eb3c3f3a3f834b9");
        sepList.add("3cedf1751c62911ad65f6e3e5b54b3d3");
        sepList.add("44a4f7fc1447b3d2f9aebe00a080ceda");
        sepList.add("44b50dcdb02bc8b16722341b80e1b5cd");
        sepList.add("464fee8b80a235ed35852a658590d9ec");
        sepList.add("467501e393df34683f6869eb7e596e0b");
        sepList.add("47034f366298c768ef863421f152bf9d");
        sepList.add("49612c3d21a4f086586e2802e5a78f5e");
        sepList.add("50f54b2251638d7ded584716f6e86b59");
        sepList.add("54d100fea11bb98d13f5398e103e812c");
        sepList.add("5e4add0832c664f50174c43ccf3ed97c");
        sepList.add("621fb5af40f292bcfb4f6c52881bb4a5");
        sepList.add("62581b9d330fc435db5f3835887998c4");
        sepList.add("63be0452b71124251e474261dabe22bf");
        sepList.add("68727551b7290cbdbba4c7fe3d246dc5");
        sepList.add("6a8dd9c0620d4dc62a0fd4fbf93f8b0f");
        sepList.add("6b1832267627ea8d3c1aacd5c5af601a");
        sepList.add("701e3cb2ce13a0f46a35abecb5ec9b28");
        sepList.add("7189ed41d48c9d7eba607d3363dabd19");
        sepList.add("727dcbb8a73af1c9bbd9ec441a265b23");
        sepList.add("76973ff13f4047a754066b25ec5aa3b8");
        sepList.add("782baeff858a011e7393ae098b2e0eda");
        sepList.add("7b07da6a94e001a4ffc37df9846d7550");
        sepList.add("7caa0e18c90b82ebf1726935f5d814d6");
        sepList.add("7d1b24b245d344efff7868789510e0c3");
        sepList.add("80365f25c39074b17239fa881485d0d3");
        sepList.add("80641a3dbb7fd94310f4122522219afb");
        sepList.add("80cdb9ccadddbad09bd3bfd34abb2a22");
        sepList.add("82b8bf46fed5d1d9496da0462a16d092");
        sepList.add("8596128a5572362f1361ad32891b60ed");
        sepList.add("87829ef9f2d09057823e9065fffc885d");
        sepList.add("897065d8069767f8f49694a0aa1cfd1c");
        sepList.add("89e2f14c09df942793e69dba3ce3afad");
        sepList.add("8e455b6d21d2745fcdbc66d0d2881d8b");
        sepList.add("8f586948e7d5807031c77e9d6ab91871");
        sepList.add("923dabf028f90c2273330d9bab4b0e60");
        sepList.add("924449aa1b6e181681f36be35ceac712");
        sepList.add("92a56c900d1077235abd61ae485037be");
        sepList.add("9ace8d828822fe2ab34dd7adfac2d52e");
        sepList.add("9f158cb622f3496fb5021056ad8b190a");
        sepList.add("a07e0369430aa84bf9cc8bda091209b5");
        sepList.add("a07f96fa337f12583695c2593d9c6e7c");
        sepList.add("a0fd03711d93bd008c34865f1d606e7e");
        sepList.add("a1f894cc8696383622f30d64b3db9e92");
        sepList.add("a30872b0fb866acef92d6c7988200bad");
        sepList.add("a436c762e203d359f5dcc1e3c7d6de2e");
        sepList.add("a46952a2091a7c62bd4796b44e0cd49f");
        sepList.add("a6b816afab08a905761e3e6fbb0e28cd");
        sepList.add("abe9eb6c90bcab798c418b525de1c127");
        sepList.add("adb5eaaa7df029dd6da09414a779de27");
        sepList.add("ae62e5d7cbdf00289b6e6451b32ab2cb");
        sepList.add("ae636892179a6ff699c0fbd0e1a03e64");
        sepList.add("b0cd8e138d1e970ca172e44f900c6564");
        sepList.add("b2558efcad6f5577febaa77c4cc73490");
        sepList.add("b6f5cc2083e8b539c73c191da146262d");
        sepList.add("ba49768e5499d6b6f2552804d47caa8c");
        sepList.add("bc07d2e58d7ce84203fb4740436a66c3");
        sepList.add("bc894a61d05e384507a5e247802404db");
        sepList.add("bcdbd32490d1e052fe15a46a95a1c074");
        sepList.add("bf54642dab8626c4e7b2d4e8b9fbda4f");
        sepList.add("bfa548f0824837e406ea33e0b42b3a99");
        sepList.add("c0e36c1c3ae0120dda83c8b0d2aa2ef0");
        sepList.add("c8f38ee2caaa1ee4543a8485d2ef7bc7");
        sepList.add("cc46edb60806a76e625ae730abc802de");
        sepList.add("d1c6ce95512a557eec646eba5e9a9dd1");
        sepList.add("d6126d1bd2b02bd7e884239e81500b57");
        sepList.add("d865f042bc7be526ff12a94018537ba6");
        sepList.add("dc2be972b5c78f7ebf52697efbecd73e");
        sepList.add("df15e2fc7356a70536d5922604348411");
        sepList.add("e2e8acd11e8e48ccdd854dfc66593476");
        sepList.add("e48de490d9b412c273ea966af4750219");
        sepList.add("ea2e1f4245153132400516ec21186dd9");
        sepList.add("eaf6e52deabff5bed59ac3475255a466");
        sepList.add("eb5e225b336d27a020ce4034221d318d");
        sepList.add("edfce06402920ca02801636df187e7f5");
        sepList.add("eedcaab0e29ef1ce3117aaf555e44280");
        sepList.add("ef3fd8480b01d2f1c124626112f1229e");//97
        sepList.add("f5ca63422ff7084adde89bbd2eac10d3");
        sepList.add("ff92cf637ae255a3c7f1bafddccc8a02");//99

        Collections.sort(sepList, comp);


    }

}
