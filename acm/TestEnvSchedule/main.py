#!/usr/bin/python3
from flask import Flask,request,make_response
import json
import sqlite3

app = Flask(__name__)

def dbaQueryScheduleList():
    conn = sqlite3.connect("./data/env.db")
    cur = conn.cursor()
    sqlstr = "select envName,fromTime, toTime, who, why from envschedule order by envName,fromTime asc"
    print(sqlstr)
    res = cur.execute(sqlstr)
    allEnv = res.fetchall()#list
    result = list()
    for e in allEnv:
        e = e #type:tuple
        env = dict()
        env["envName"] = e[0]
        env["fromTime"] = e[1]
        env["toTime"] = e[2]
        env["who"] = e[3]
        env["why"] = e[4]
        result.append(env)
    conn.close()
    return json.dumps(result)

def dbaCheckPassword(password):
    conn = sqlite3.connect("./data/env.db")
    cur = conn.cursor()
    sqlstr = "select passwd from superuser where passwd='{}'".format(password)
    print(sqlstr)
    res = cur.execute(sqlstr)
    allUser = res.fetchall()#list
    conn.close()
    if len(allUser) > 0:
        return True
    return False

def dbaQueryAvailableEnv(fromTime, toTime):
    conn = sqlite3.connect("./data/env.db")
    cur = conn.cursor()
    sqlstr = "select envName from envinfo"
    print(sqlstr)
    res = cur.execute(sqlstr)
    allEnv = res.fetchall()#list
    print(allEnv)
    sqlstr = "select envName from envschedule where  toTime >='{}' and toTime <='{}' or fromTime >= '{}' and fromTime <= '{}' order by envName asc".format(fromTime, toTime, fromTime, toTime)
    print(sqlstr)
    res = cur.execute(sqlstr)
    usedEnv = res.fetchall() #list
    for e in usedEnv:
        allEnv.remove(e)
    print(allEnv)
    result = list()
    for e in allEnv:
        e = e #type:tuple
        env = dict()
        env["envName"] = e[0]
        result.append(env)
    conn.close()
    return json.dumps(result)

def dbaBookEnv(fromTime, toTime, envName, who, why):
    conn = sqlite3.connect("./data/env.db")
    sqlstr = "select envName from envinfo where  envName='{}'".format(envName)
    print(sqlstr)
    res = conn.cursor().execute(sqlstr)
    usedEnv = res.fetchall() #list
    if len(usedEnv) < 1:
        conn.close()
        return '{"error":1001, "msg":"invalid environment name"}'
    sqlstr = "delete  from envschedule where envName='{}' and fromTime='{}' and toTime='{}' and who='{}'".format(envName, fromTime, toTime, who)
    print(sqlstr)
    res = conn.execute(sqlstr)
    sqlstr = "insert into envschedule(envName, fromTime, toTime, who, why) values('{}','{}','{}', '{}', '{}')".format(envName, fromTime, toTime, who, why)
    print(sqlstr)
    res = conn.execute(sqlstr)
    conn.commit()
    conn.close()
    return '{"msg":"book success", "error":0}'

def dbaDeleteScheduleInfo(fromTime, toTime, envName):
    conn = sqlite3.connect("./data/env.db")
    sqlstr = "delete  from envschedule where envName='{}' and fromTime='{}' and toTime='{}'".format(envName, fromTime, toTime)
    print(sqlstr)
    res = conn.execute(sqlstr)
    conn.commit()
    conn.close()
    return '{"msg":"delete success", "error":0}'




@app.route('/queryAvailableEnv', methods=['POST'])
def queryAvailableEnv():
    inputStr = str(request.stream.read(), encoding='utf8')
    print(inputStr)
    jsonObj = json.loads(inputStr) #type:dict
    fromTime = jsonObj.get("fromTime")
    toTime = jsonObj.get("toTime")
    result = dbaQueryAvailableEnv(fromTime, toTime)
    return result

@app.route('/bookEnv', methods=['POST'])
def bookEnv():
    inputStr = str(request.stream.read(), encoding='utf8')
    print(inputStr)
    jsonObj = json.loads(inputStr) #type:dict
    fromTime = jsonObj.get("fromTime")
    toTime = jsonObj.get("toTime")
    envName = jsonObj.get("envName")
    who = jsonObj.get("who")
    why = jsonObj.get("why")
    password = jsonObj.get("password")

    result = dbaCheckPassword(password)
    if not result:
        resp = make_response('{"msg":"invalid password", "error":1001}') 
        resp.delete_cookie("testenvschedule_p")
        return resp
    result = dbaBookEnv(fromTime, toTime, envName, who, why)
    return result

@app.route('/queryScheduleList', methods=['POST'])
def queryScheduleList():
    result = dbaQueryScheduleList()
    return result

@app.route('/deleteScheduleInfo', methods=['POST'])
def deleteScheduleInfo():
    inputStr = str(request.stream.read(), encoding='utf8')
    print(inputStr)
    jsonObj = json.loads(inputStr) #type:dict
    fromTime = jsonObj.get("fromTime")
    toTime = jsonObj.get("toTime")
    envName = jsonObj.get("envName")
    password = jsonObj.get("password")

    result = dbaCheckPassword(password)
    if not result:
        resp = make_response('{"msg":"invalid password", "error":1001}') 
        resp.delete_cookie("testenvschedule_p")
        return resp
    
    result = dbaDeleteScheduleInfo(fromTime, toTime, envName)
    return result

@app.route('/checkPassword', methods=['POST'])
def checkPassword():
    inputStr = str(request.stream.read(), encoding='utf8')
    print(inputStr)
    jsonObj = json.loads(inputStr) #type:dict
    password = jsonObj.get("password")
    
    result = dbaCheckPassword(password)
    if result:
        resp = make_response('{"msg":"login success", "error":0}') 
        resp.set_cookie("testenvschedule_p", password)
        return resp
    else:
        resp = make_response('{"msg":"invalid password", "error":1001}') 
        resp.delete_cookie("testenvschedule_p")
        return resp

if __name__ == '__main__':
    app.run("0.0.0.0", 8080)

