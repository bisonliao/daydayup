package org.apache.dubbo.demo.provider;

import org.apache.dubbo.demo.intf.DemoService;

public class DemoServiceImpl implements DemoService {
    public String sayHello(String name) {
        return "Hello " + name;
    }
}

