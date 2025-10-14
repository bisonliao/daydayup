import torch
import torch.nn as nn
import torch.nn.functional as F
from conf.frap_config import Config

'''
class Config:
    """配置参数类"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 交通运动数量（8个方向）
    MOVEMENT_NUM = 8
    
    # 相位数量（4个相位）
    PHASE_NUM = 4
    
    # 每个相位对应的两个运动ID（根据论文图3的相位定义）
    # 格式: [phase_id] = [movement_id1, movement_id2]
    PHASE_MOVEMENTS = [
        [2, 6],
        [0, 4],
        [3, 7],
        [1, 5],
    ]
    
    # 网络维度配置
    DEMAND_HIDDEN_DIM = 32
    DEMAND_OUTPUT_DIM = 16
    RELATION_EMBEDDING_DIM = 16
    DEMAND_EMBEDDING_DIM = 32
    COMPETITION_HIDDEN_DIM = 64
    COMPETITION_LAYERS = 2

'''

class FRAPModel(nn.Module):
    """
    FRAP模型 - 基于相位竞争的交通信号控制神经网络
    
    根据论文描述，FRAP模型包含三个阶段：
    1. 相位需求建模 (Phase Demand Modeling)
    2. 相位对嵌入 (Phase Pair Embedding)
    3. 相位对竞争 (Phase Pair Competition)
    """
    
    def __init__(self):
        super(FRAPModel, self).__init__()
        
        
        # 阶段1: 相位需求建模
        # 每个交通运动的车辆数量和信号状态作为输入
        self.demand_net = nn.Sequential(
            nn.Linear(2, Config.DEMAND_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.DEMAND_HIDDEN_DIM, Config.DEMAND_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.DEMAND_HIDDEN_DIM, Config.DEMAND_OUTPUT_DIM)
        )
        
        # 阶段2: 相位对关系嵌入
        # 创建相位冲突矩阵（根据论文图3(d)）
        self.relation_matrix = self._create_relation_matrix() #[PHASE_NUM, PHASE_NUM]
        # 2表示两种关系：有共同的mv和没有共同的mv。nn.Embedding本质上是查表。本环境中实际情况就是只会查idx=0的这个表项，因为相位间都没有共同的mv
        self.relation_embedding = nn.Embedding(2, Config.RELATION_EMBEDDING_DIM) # [PHASE_NUM, PHASE_NUM, RELATION_EMBEDDING_DIM]
        
        # 阶段3: 相位对竞争网络
        # 使用多个全连接层代替1x1卷积
        self.competition_layers = nn.ModuleList()
        for i in range(Config.COMPETITION_LAYERS):
            if i == 0:
                in_dim = Config.DEMAND_EMBEDDING_DIM
            else:
                in_dim = Config.COMPETITION_HIDDEN_DIM
            
            self.competition_layers.append(nn.Linear(in_dim, Config.COMPETITION_HIDDEN_DIM))

        self.relation_layers = nn.ModuleList()
        for i in range(Config.RELATION_LAYERS):
            if i == 0:
                in_dim = Config.RELATION_EMBEDDING_DIM
            else:
                in_dim = Config.COMPETITION_HIDDEN_DIM
            
            self.relation_layers.append(
                nn.Linear(in_dim, Config.COMPETITION_HIDDEN_DIM)
            )
        
        # 最终竞争结果层
        self.competition_output = nn.Linear(Config.COMPETITION_HIDDEN_DIM, 1)
        
    def _create_relation_matrix(self):
        """
        创建相位关系矩阵
        返回值: 形状为 [PHASE_NUM, PHASE_NUM] 的矩阵
                0表示不冲突，1表示冲突
        """
        relation_matrix = torch.zeros(
            (Config.PHASE_NUM, Config.PHASE_NUM), 
            dtype=torch.long,
            device=Config.DEVICE
        )
        
        # 简化实现：假设不同相位的运动有重叠即为冲突
        for i in range(Config.PHASE_NUM):
            for j in range(Config.PHASE_NUM):
                if i == j:
                    continue  # 同一相位不比较
                
                movements_i = set(Config.PHASE_MOVEMENTS[i])
                movements_j = set(Config.PHASE_MOVEMENTS[j])
                
                # 如果有共同运动，则冲突
                if movements_i & movements_j:
                    relation_matrix[i, j] = 1
        
        return relation_matrix
    
    def forward(self, mv_veh_cnt, phase_onehot):
        """
        前向传播
        
        参数:
            mv_veh_cnt: 每个交通运动的车辆数量 [batch_size, MOVEMENT_NUM, 1]
            phase_onehot: 当前相位one-hot编码 [batch_size, PHASE_NUM]
        
        返回:
            Q_values: 每个相位的Q值 [batch_size, PHASE_NUM]
        """
        batch_size = mv_veh_cnt.size(0)
        
        # 将相位one-hot转换为运动级别的绿灯状态
        movement_green = self._phase_to_movement_green(phase_onehot)  # [batch_size, MOVEMENT_NUM]
        movement_green = movement_green.unsqueeze(-1)  # [batch_size, MOVEMENT_NUM, 1]
        
        # 阶段1: 相位需求建模
        # Linear内部会先把输入张量临时 reshape 成二维矩阵，再做矩阵乘法，也就是虽然输入输出是多维的tensor，但Linear只对最后一维做映射变换
        # 这个是FRAP工作的关键，没有针对每个movement独特的参数，所有movement都共享一组参数，该参数把输入2个映射为输出 DEMAND_OUTPUT_DIM个
        # 后面的神经网络层也是类似的。
        movement_features = torch.cat([mv_veh_cnt, movement_green], dim=-1)  # [batch_size, MOVEMENT_NUM, 2]
        movement_demands = self.demand_net(movement_features)  # [batch_size, MOVEMENT_NUM, DEMAND_OUTPUT_DIM]
        
        # 计算每个相位的需求（相加对应两个运动的需求）
        phase_demands = torch.zeros(
            batch_size, Config.PHASE_NUM, Config.DEMAND_OUTPUT_DIM,
            device=Config.DEVICE
        )
        
        for phase_id in range(Config.PHASE_NUM):
            mov1, mov2 = Config.PHASE_MOVEMENTS[phase_id]
            phase_demands[:, phase_id, :] = (
                movement_demands[:, mov1, :] + movement_demands[:, mov2, :]
            )
        
        # 阶段2: 相位对嵌入
        # 创建所有相位对的关系嵌入 和需求嵌入 两个立方体

        # 输入 [PHASE_NUM, PHASE_NUM, RELATION_EMBEDDING_DIM]， 输出[PHASE_NUM, PHASE_NUM, RELATION_EMBEDDING_DIM]
        relation_embeddings = self.relation_embedding(self.relation_matrix)  # [PHASE_NUM, PHASE_NUM, RELATION_EMBEDDING_DIM]
        
        # 扩展关系嵌入到批次维度, 得到关系嵌入立方体
        relation_volume = relation_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )  # [batch_size, PHASE_NUM, PHASE_NUM, RELATION_EMBEDDING_DIM]
        
        # 创建需求嵌入立方体
        demand_volume = torch.zeros(
            (batch_size, Config.PHASE_NUM, Config.PHASE_NUM, Config.DEMAND_EMBEDDING_DIM),
            device=Config.DEVICE
        )
        
        for i in range(Config.PHASE_NUM):
            for j in range(Config.PHASE_NUM):
                if i != j:
                    # 连接两个相位的需求表示
                    demand_volume[:, i, j, :] = torch.cat([
                        phase_demands[:, i, :], 
                        phase_demands[:, j, :]
                    ], dim=-1)
        
        # 阶段3: 相位对竞争
                
        # 应用竞争网络， 一层一层执行1x1卷积
        # 利用了Linear只对最后一个维度做变换的特性，前面的维度保持不变，相当于1x1的卷积。
        competition_output = demand_volume
        for layer in self.competition_layers:
            competition_output = F.relu(layer(competition_output))

        relation_output = relation_volume
        for layer in self.relation_layers:
            relation_output = F.relu(layer(relation_output))
        
        competition_output = competition_output * relation_output # 按元素乘
        
        # 获取竞争结果，再次1x1卷积
        competition_scores = self.competition_output(competition_output).squeeze(-1)  # [batch_size, PHASE_NUM, PHASE_NUM]
        
        # 对每个相位，汇总它与所有其他相位的竞争得分
        # 对角线元素（自己与自己竞争）设为0
        mask = torch.eye(
            Config.PHASE_NUM, 
            device=Config.DEVICE
        ).bool()
        competition_scores.masked_fill_(mask.unsqueeze(0), 0)
        
        # 计算每个相位的总得分
        Q_values = torch.sum(competition_scores, dim=-1)  # [batch_size, PHASE_NUM]
        
        return Q_values
    
    def _phase_to_movement_green(self, phase_onehot):
        """
        将相位one-hot编码转换为运动级别的绿灯状态
        
        参数:
            phase_onehot: 当前相位one-hot编码 [batch_size, PHASE_NUM]
        
        返回:
            movement_green: 每个运动是否绿灯 [batch_size, MOVEMENT_NUM]
        """
        batch_size = phase_onehot.size(0)
        movement_green = torch.zeros(
            batch_size, Config.MOVEMENT_NUM,
            device=Config.DEVICE
        )
        
        # 对于每个相位，将其对应的两个运动设置为绿灯
        for phase_id in range(Config.PHASE_NUM):
            mov1, mov2 = Config.PHASE_MOVEMENTS[phase_id]
            phase_active = phase_onehot[:, phase_id].unsqueeze(-1)  # [batch_size, 1]
            
            movement_green[:, mov1] += phase_active.squeeze()
            movement_green[:, mov2] += phase_active.squeeze()
        
        # 确保没有运动被多个相位设置为绿灯（理论上不应该发生）
        movement_green = torch.clamp(movement_green, 0, 1)
        
        return movement_green


