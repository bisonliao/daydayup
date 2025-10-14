
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import datetime
import os
from env.TscEnv import TscEnv
from torch.utils.tensorboard import SummaryWriter
from conf.simple_config import Config
from env.TscEnv import TscEnv
from model.simple_model import DQN
from tools.common import *


'''
特征里，每个方向上的排队长度要安排上
出现了没有车通过，也一直亮着绿灯的情况。所以最大相位时长要做限制。最小的也有必要，虽然学习到了较长的相位，但还是有个别相位时间很短
某个队列长度超过阈值，就终止回合，并做较大的处罚

有的方向上车辆不多，就会一直等一直等。解决方法：各个方向上的排队最大时长要输入，并针对多次达到最大阈值，给与较大的处罚；或者奖励除了是等待车辆数的负数，也是等待时长的负数？两者的数量级要对等才行
'''


class SimpleAgent:
    def __init__(self):
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(f'logs/tsc_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}')
        # 超参数
        self.gamma = 0.9  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最低探索率
        self.epsilon_decay = 0.97  # 探索率衰减
        self.learning_rate = 1e-3  # 学习率
        self.batch_size = 128  # 经验回放的批量大小
        self.memory_size = 100000  # 经验池大小
        self.target_update_freq = 3  # 目标网络更新频率

        self.env = TscEnv(self.writer)
        self.n_state = self.env.observation_space.shape[0]  # 状态维度
        self.n_action = self.env.action_space.n  # 动作数量
        print(f'action number={self.n_action}, obs shape={self.env.observation_space.shape}')

        # 初始化网络
        self.model = DQN(self.n_state, self.n_action).to(self.device)
        self.target_model = DQN(self.n_state, self.n_action).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=self.memory_size)

        self.long_q_duration = 0
        self.total_step = 0
        self.action_history = deque(maxlen=100)

    def reset(self):
        self.long_q_duration = 0


    def select_action(self, state, epsilon):
        """基于 ε-greedy 选择动作"""
        if random.random() < epsilon:
            return random.randint(0, self.n_action - 1)  # 随机选择
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 变换前：[4] -> 变换后：[1, 4]
            action = self.model(state).argmax(1).item()  # 选取 Q 值最大的动作
            self.action_history.append(action)
            return action


    def train(self):
        if len(self.memory) < self.batch_size:
            return 9999.0  # 经验池数据不足时不训练

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)  # (batch_size, 4)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # (batch_size,) -> (batch_size, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # (batch_size,) -> (batch_size, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)  # (batch_size, 4)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # (batch_size,) -> (batch_size, 1)

        # 计算当前 Q 值
        q_values = self.model(states).gather(1, actions)  # 从 Q(s, a) 选取执行的动作 Q 值

        # 计算目标 Q 值
        next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]  # 选取 Q(s', a') 的最大值
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)  # TD 目标

        # 计算损失
        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self.total_step % 901 == 0:
            log_network_statistics(self.model, self.writer,self.total_step)
            check_grad_norm(self.model, self.writer, self.total_step)
        self.optimizer.step()

        return loss.item()


    def save_checkpoint(self, id):
        os.makedirs('./checkpoints/', exist_ok=True)
        path=f"./checkpoints/dqn_checkpoint_{id}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")


    def load_checkpoint(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Checkpoint loaded from {path}")
        else:
            print("No checkpoint found, starting from scratch.")

    def observation_process(self):
        self.env.calc_vehicle_info()

        # 当前相位的one-hot表示
        phase_onehot = np.zeros(Config.PHASE_NUM, dtype=np.long)
        phase_onehot[self.env.current_phase-1] = 1
               

        # 合并所有状态信息
        state = np.concatenate(
            [
            phase_onehot,
            [self.env.phase_duration / Config.MAX_PHASE_DUR],
            self.env.vehicle_dense, 
            self.env.avg_speed / 22,
            self.env.waiting_qlen / Config.MAX_QLEN,
     
            ], 
            dtype=np.float32)

        return state
    def reward_shape(self):

        # 某些方向的等待车辆堆积超过阈值,
        if np.any(self.env.waiting_qlen >= Config.MAX_QLEN):
            self.long_q_duration += 1
        else:
            self.long_q_duration = 0

        if self.long_q_duration > 20:
            #print(f'too long queue')
            return -10
        
        # 某个相位持续时间超过 30s
        if self.env.phase_duration >= Config.MAX_PHASE_DUR:  
            #print(f'too long phase')
            return -10
        

        vehicle_cnt = np.sum(self.env.waiting_qlen)

        if self.env.phase_duration == 1: #刚经历了切换，有3s红灯时间成本，所以-1处罚
            reward = -(vehicle_cnt**0.25) - 1
        else:
            reward = -(vehicle_cnt**0.25)

        return reward


    def learn_or_eval(self, mode):
        
        if mode == "train":
            episodes = 1000

            '''if input("preload model?[Y/n]") == "Y":
                self.load_checkpoint('./checkpoints/dqn_checkpoint_-20657.283203125.pth')
                self.epsilon = 0.1
                print(f'preload successfully')'''

            for episode in range(episodes):
                save_replay = ( episode > 20 and episode % 37 == 7 )

                self.reset()
                self.env.reset(save_replay=save_replay)
                state = self.observation_process()
                total_reward = 0

                while True:
                    action = self.select_action(state, self.epsilon)
                    _, _, terminated, truncated, _ = self.env.step(action)
                    self.total_step += 1
                    next_state = self.observation_process()
                    reward = self.reward_shape()
                    done = terminated or truncated
                    if self.total_step % 117 == 7:
                        self.writer.add_scalar('train/step_reward', reward, self.total_step)

                    
                    # 经验回放缓存
                    self.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    total_reward += reward

                    # 训练 DQN
                    loss = self.train()

                    if done:
                        break

                # 逐步降低 epsilon，减少随机探索，提高利用率
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                # 定期更新目标网络，提高稳定性
                if episode % self.target_update_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                # 定期保存模型
                if episode % 25 == 1:
                    self.save_checkpoint(f'{episode}_{total_reward:.3f}')

                att,thrput = self.env.get_biz_metrics()

                print(f"{datetime.datetime.now().strftime('%H:%M:%S')} Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}, loss:{loss:.8f}")
                self.writer.add_scalar('train/episode_reward', total_reward, episode)
                self.writer.add_scalar('train/epsilon', self.epsilon, episode)
                self.writer.add_scalar('train/loss', loss, episode)
                self.writer.add_scalar('train/att', att, episode)
                self.writer.add_scalar('train/thrput', thrput, episode)
                self.writer.add_scalar('train/action_entropy', check_action_entropy(self.action_history), episode)

        else:
            self.load_checkpoint('./checkpoints/dqn_checkpoint_126_-4186.705.pth')
            self.model.eval()
            eval_env = TscEnv(self.writer)
            self.env = eval_env

            eval_env.reset(save_replay=True)
            state = self.observation_process()

            while True:
                action = self.select_action(state, 0)
                _, _, terminated, truncated, _ = eval_env.step(action)
                next_state = self.observation_process()
                done = terminated or truncated
                state = next_state
                    
                if done:
                    break
            att,thrput = self.env.get_biz_metrics()
            print(f'att={att}, thruput={thrput}')

if __name__ == "__main__":
    agent = SimpleAgent()
    agent.learn_or_eval('eval')
    
    
