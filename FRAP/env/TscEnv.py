import datetime
import random
import time

import numpy as np
import torch
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter
import math
import cityflow
from collections import deque
from conf.simple_config import Config



class TscEnv(gym.Env):
   
    def __init__(self, writer:SummaryWriter):

        super(TscEnv, self).__init__()
        self.writer = writer

        self.eng = cityflow.Engine(f"{Config.ENV_BASE_DIR}/config.json", thread_num=4)
        
        self.vehicles_ever_seen = set() # 记录凡是出现过的车辆
       
        # 定义动作空间和状态空间
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low= np.array([0,0,0,0,  0,  0,0,0,0,  0,0,0,0, 0,0,0,0]),
            high=np.array([1,1,1,1,  1,  1,1,1,1,  1,1,1,1, 1,1,1,1]),
            dtype=np.float32
        )
        

        # 环境参数
        self.max_steps = 2000  # 最大步数
        self.current_step = 0 #每一回合里的步数计数器
        self.total_step = 0  #环境运行过程中一直累加的总的计数器
        self.durations = deque(maxlen=100) #统计过去100个相位的时间长度

        self.reset()

    

    def reset(self, seed=None, save_replay=False):
        """重置环境到初始状态"""
        #self.eng.set_save_replay(save_replay)
        #self.eng.set_replay_file(f'replay_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}.txt')
        self.eng.reset()
        self.eng.set_tl_phase('intersection_1_1', 1)
        self.current_phase = 1

        self.current_step = 0
        self.phase_duration = 1 # 一个相位持续多久
        self.vehicles_ever_seen.clear()
        if len(self.durations) > 0:
            self.writer.add_scalar('phase_duration/max', np.max(self.durations), self.total_step)
            self.writer.add_scalar('phase_duration/min', np.min(self.durations), self.total_step)
            self.writer.add_scalar('phase_duration/avg', np.mean(self.durations), self.total_step)
            self.writer.add_scalar('phase_duration/std', np.std(self.durations), self.total_step)
        self.durations.clear()
        
        # 获取初始状态
        state = self._get_state()

        return state,{}

    def step(self, action, dur=1):

        new_phase = action + 1

        if self.current_phase != new_phase: #相位有变化
            # 先来3s红灯清场
            self.eng.set_tl_phase('intersection_1_1', 0)
            for _ in range(3):
                self.eng.next_step()
            # 然后轮转到下一个方向放行
            self.eng.set_tl_phase('intersection_1_1',new_phase)
            self.current_phase = new_phase

            self.durations.append(self.phase_duration)
            self.phase_duration = 0
        
        
        for _ in range(dur):
            self.eng.next_step()
            self.phase_duration += 1


        self.current_step += 1
        self.total_step += 1

        
        # 获取新状态
        state = self._get_state()

        # 计算奖励
        reward = 0
        terminated = False


        # 检查是否超过最大步数
        truncated = False
        if self.eng.get_current_time() >= self.max_steps:
            truncated = True


        info = {
            "steps": self.current_step,
        }


        return state, reward, terminated, truncated, info
    
    def _route2phaseID(self, route:str):
        route = route.strip()
        if route == "road_2_1_2 road_1_1_2" or route == "road_0_1_0 road_1_1_0": # 东西直行
            return 0
        if route == "road_1_2_3 road_1_1_3" or route == "road_1_0_1 road_1_1_1": #南北直行
            return 1
        if route == "road_2_1_2 road_1_1_3" or route == "road_0_1_0 road_1_1_1": # 东西左转
            return 2
        if route == "road_1_2_3 road_1_1_0" or route == "road_1_0_1 road_1_1_2": # 南北左转
            return 3
        assert False, f"invalid route {route} "

    def _route2movement(self, route:str):
        route = route.strip()
        if route == "road_2_1_2 road_1_1_2":
            return 2
        if route == "road_0_1_0 road_1_1_0": # 东西直行
            return 6
        if route == "road_1_2_3 road_1_1_3" :
            return 0
        if route == "road_1_0_1 road_1_1_1": #南北直行
            return 4
        if route == "road_2_1_2 road_1_1_3" :
            return 3
        if route == "road_0_1_0 road_1_1_1": # 东西左转
            return 7
        if route == "road_1_2_3 road_1_1_0" : 
            return 1
        if route == "road_1_0_1 road_1_1_2": # 南北左转
            return 5
        assert False, f"invalid route {route} "


    def calc_vehicle_info(self):

        dic_lane_vehicles = self.eng.get_lane_vehicles()
        vehicles_present = set()

        self.vehicle_dense = np.zeros(Config.PHASE_NUM, dtype=np.float32) #停止线前面30m范围内的车辆密度
        self.avg_speed = np.zeros(Config.PHASE_NUM, dtype=np.float32)#停止线前面30m范围内的车辆平均速度
        self.waiting_qlen = np.zeros(Config.PHASE_NUM, dtype=np.float32) # 入口车道上排队的车辆数目，由于每个方向都是一个入口车道，所以这里就是等待队列的长度


        speed_list = [deque(maxlen=100) for _ in range(Config.PHASE_NUM)]
        in_roads = {'road_2_1_2', 'road_0_1_0', 'road_1_2_3', 'road_1_0_1'}
        out_roads = {'road_1_1_2', 'road_1_1_0', 'road_1_1_3', 'road_1_1_1'}
        for lane_id in dic_lane_vehicles:
            road_id = lane_id[:-2]

            if road_id not in in_roads and road_id not in out_roads:
                continue

            vehicle_ids = dic_lane_vehicles[lane_id]

            for vid in vehicle_ids:
                try:
                    info = self.eng.get_vehicle_info(vid)
                    if info.get('running', "0") != "1": #whether the vehicle is running
                        continue

                    #  The next intersection if the vehicle is running on a lane
                    '''if info.get('intersection', "")!= 'intersection_1_1':
                        continue'''

                    self.vehicles_ever_seen.add(vid)

                    if road_id not in in_roads: #不在入口车道里
                        continue

                    ##-------- 这一步往后，都是在intersection_1_1的入口车道上 ----#
                    vehicles_present.add(vid)
                    

                    speed = info['speed']       
                    route = info['route'] # A string contains ids of following roads in the vehicle’s route which are separated by space
                    phase_id = self._route2phaseID(route)

                    if float(speed) < 0.1:
                        self.waiting_qlen[phase_id] += 1

                    drivable = info['drivable']
                    vehicle_distance = info["distance"]  # 已行驶距离
                    to_stop_line = Config.LANE_LEN - float(vehicle_distance)
                    if to_stop_line > (Config.CELL_LEN *Config.OBSERV_CELL_NUM) or to_stop_line < 0:
                        continue

                    
                    self.vehicle_dense[phase_id] += 1.0 / (Config.OBSERV_CELL_NUM * 2) 
                    speed_list[phase_id].append( float(speed) )

                except Exception as e:
                    print(f"Error retrieving info for vehicle {vid}: {e}")
            
        for i in range(Config.PHASE_NUM):
            if len(speed_list[i]) > 0:
                self.avg_speed[i] = sum(speed_list[i]) / len(speed_list[i])

        self.throughput = len(self.vehicles_ever_seen - vehicles_present) #从回合开始（reset后）到目前，通过该路口的车辆数，即吞吐量


    def calc_vehicle_info2(self):
        
        vehicles_present = set()

        self.vehicle_dense = np.zeros(Config.PHASE_NUM, dtype=np.float32) #停止线前面30m范围内的车辆密度
        self.avg_speed = np.zeros(Config.PHASE_NUM, dtype=np.float32)#停止线前面30m范围内的车辆平均速度
        self.waiting_qlen = np.zeros(Config.PHASE_NUM, dtype=np.float32) # 入口车道上排队的车辆数目，由于每个方向都是一个入口车道，所以这里就是等待队列的长度


        speed_list = [deque(maxlen=100) for _ in range(Config.PHASE_NUM)]
        vehicle_ids = self.eng.get_vehicles(True)
        for vid in vehicle_ids:
            try:
                info = self.eng.get_vehicle_info(vid)
                if info.get('running', "0") != "1": #whether the vehicle is running
                    continue

                #  The next intersection if the vehicle is running on a lane
                if info.get('intersection', "")!= 'intersection_1_1':
                    continue

                self.vehicles_ever_seen.add(vid)
                vehicles_present.add(vid)
                

                speed = info['speed']       
                route = info['route'] # A string contains ids of following roads in the vehicle’s route which are separated by space
                phase_id = self._route2phaseID(route)

                if float(speed) < 0.1:
                    self.waiting_qlen[phase_id] += 1

                drivable = info['drivable']
                vehicle_distance = info["distance"]  # 已行驶距离
                to_stop_line = Config.LANE_LEN - float(vehicle_distance)
                if to_stop_line > (Config.CELL_LEN *Config.OBSERV_CELL_NUM) or to_stop_line < 0:
                    continue

                
                self.vehicle_dense[phase_id] += 1.0 / (Config.OBSERV_CELL_NUM * 2) 
                speed_list[phase_id].append( float(speed) )

            except Exception as e:
                print(f"Error retrieving info for vehicle {vid}: {e}")
        
        for i in range(Config.PHASE_NUM):
            if len(speed_list[i]) > 0:
                self.avg_speed[i] = sum(speed_list[i]) / len(speed_list[i])

        self.throughput = len(self.vehicles_ever_seen - vehicles_present) #从回合开始（reset后）到目前，通过该路口的车辆数，即吞吐量

    def _get_state(self):
        return None

    
    def get_biz_metrics(self): 
        att = self.eng.get_average_travel_time()
        thput =self.throughput
        return att, thput


    def render(self, mode='human'):
        """渲染环境"""
        pass  # PyBullet会自动处理渲染

    def close(self):
        """关闭环境"""
        pass
