#!/usr/bin/python3

import json
import sqlite3

conn = sqlite3.connect("./data/env.db")
#conn.execute("create table envinfo(envName char(128))")
#conn.execute("create table envschedule(envName char(128), fromTime datetime, toTime datetime, who char(128), why char(256))")
#conn.execute("create index idx_envinfo_envname on envinfo(envName)")
#conn.execute("create index idx_envschedule_envname on envschedule(envName)")
#conn.execute("insert into  envinfo(envName) values('gugong')")
#conn.execute("insert into  envinfo(envName) values('huangpu')")
#conn.execute("insert into  envinfo(envName) values('tiantan')")
#conn.execute("delete from envschedule")
conn.commit()
print(conn.cursor().execute("select envName from envinfo where  envName='gugong'").fetchall())

