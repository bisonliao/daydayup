import torch
from conf.frap_config import Config

class Network(torch.nn.Module):
    def __init__(self, num_lanelink, num_phase, phase_2_passable_lanelink):
        super(Network, self).__init__()
        self.num_lanelink = num_lanelink
        self.num_phase = num_phase
        self.phase_2_passable_lanelink = phase_2_passable_lanelink.to(Config.DEVICE)
        self.lanelink_2_applicable_phase = phase_2_passable_lanelink.permute(1, 0).to(Config.DEVICE)  # num_lanelink * num_phase
        self.phase_competition_mask = self._get_phase_competition_mask(phase_2_passable_lanelink)  # num_phase * (num_phase - 1)
        self.dim_embedding = 4
        self.dim_hidden_repr = 16
        self.dim_conv_repr = 20  # section 4.3.5

        
        self.phase_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.dim_embedding),
            torch.nn.ReLU(),
        )  # Ws, bs in Eq. 3
        self.num_vehicle_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.dim_embedding),
            torch.nn.ReLU(),
        )  # Wv, bv in Eq. 3
        self.lanelink_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.dim_embedding * 2, out_features=self.dim_hidden_repr),
            torch.nn.ReLU(),
        )  # Wh, bh in Eq. 4
        # embedding层是一个就是一个随机初始化的待学习的矩阵，然后用输入的离散的整数作为下标查表得到其中的一个向量作为输出
        self.relation_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.dim_embedding)  # sec. 4.3.3

        self.conv_cube = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.dim_hidden_repr * 2, out_channels=self.dim_conv_repr, kernel_size=(1, 1)),
            torch.nn.ReLU(),
        )
        self.conv_relation = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_conv_repr, kernel_size=(1, 1)),
            torch.nn.ReLU(),
        )
        self.tail_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.dim_conv_repr, out_channels=self.dim_conv_repr, kernel_size=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.dim_conv_repr, out_channels=1, kernel_size=(1, 1)),
        )

    def forward(self, mv_tensor, phase_tensor):
        batch_size = mv_tensor.shape[0]
        b_current_passable_lanelink = torch.matmul(phase_tensor.float(), self.phase_2_passable_lanelink.float())  # batch * num_lanelink
        b_num_waiting_vehicle = mv_tensor.squeeze(-1)  # batch * num_lanelink

        # Eq. 3
        # Linear内部会先把输入张量临时 reshape 成二维矩阵，再做矩阵乘法，也就是虽然输入输出是多维的tensor，但Linear只对最后一维做映射变换
        # 这个是FRAP工作的关键，没有针对每个movement独特的参数，所有movement都共享一组参数，该参数把输入1个映射为输出embedding_dim个
        # 后面的神经网络层也是类似的。
        lanelink_embedding_wrt_phase = self.phase_embedding(b_current_passable_lanelink.unsqueeze(-1))  # batch * num_lanelink * embedding_dim
        lanelink_embedding_wrt_num_vehicle = self.num_vehicle_embedding(b_num_waiting_vehicle.unsqueeze(-1))  # batch * num_lanelink * embedding_dim

        # Eq. 4， 
        # 得到movement层面的demand嵌入
        lanelink_embedding = torch.cat([lanelink_embedding_wrt_phase, lanelink_embedding_wrt_num_vehicle], dim=2)  # batch * num_lanelink * (2 *embedding_dim)
        lanelink_embedding = self.lanelink_embedding(lanelink_embedding)  # batch * num_lanelink * dim_hidden_repr
        # 变换一下，方便下一步的矩阵乘法
        lanelink_embedding = lanelink_embedding.permute(0, 2, 1)  # batch * dim_hidden_repr * num_lanelink

        # Eq. 5  
        # 乘以一个形为 num_lanelink * num_phase 的矩阵
        phase_embedding = torch.matmul(lanelink_embedding, self.lanelink_2_applicable_phase.float())  # batch * dim_hidden_repr * num_phase
        # 变换回来，就得到了相位的demand嵌入
        phase_embedding = phase_embedding.permute(0, 2, 1)  # batch * num_phase * dim_hidden_repr
        # 相位之间竞争的嵌入，也就是把一对相位的demand拼接在一起， 并把嵌入放在第二个维度上，作为通道，方便做二维卷积
        phase_embedding_cube = self._get_phase_embedding_cube(batch_size, phase_embedding)  # batch * (dim_hidden_repr * 2) * num_phase * (num_phase - 1)
        # Eq. 6 
        # 在相位之间竞争关系上 执行1x1卷积， 通道数发生变化，长宽和batch不变
        phase_conv = self.conv_cube(phase_embedding_cube)  # batch * dim_conv_repr * num_phase * (num_phase - 1)

        # Eq. 7
        # 相位pair的嵌入，并执行1x1卷积， 通道数发生变化，长宽和batch不变
        relation_embedding = self.relation_embedding(self.phase_competition_mask).permute(2, 0, 1).unsqueeze(0)  # (batch=1) * embedding_dim * num_phase * (num_phase - 1)
        relation_conv = self.conv_relation(relation_embedding)  # batch * dim_conv_repr * num_phase * (num_phase - 1)

        # Eq. 8
        # 两个1x1卷积的结果，按元素乘
        combined_feature = phase_conv * relation_conv  # batch * dim_conv_repr * num_phase * (num_phase - 1)
        # 再次1x1卷积， 通道数发生变化，长宽和batch不变
        before_merge = self.tail_layer(combined_feature)  # batch * 1 * num_phase * (num_phase - 1)
        q_values = torch.sum(before_merge, dim=3).squeeze(1)  # batch * num_phase
        return q_values

    def _get_phase_embedding_cube(self, batch_size, phase_embedding):
        phase_embedding_cube = torch.zeros((batch_size, 32, self.num_phase, self.num_phase - 1)).to(Config.DEVICE)
        for phase_idx in range(self.num_phase):
            continuous_jdx = 0
            for phase_jdx in range(self.num_phase):
                if phase_idx == phase_jdx:
                    continue
                phase_embedding_comb = torch.cat([
                    phase_embedding[:, phase_idx, :],
                    phase_embedding[:, phase_jdx, :]
                ], dim=1)
                phase_embedding_cube[:, :, phase_idx, continuous_jdx] = phase_embedding_comb
                continuous_jdx += 1
        return phase_embedding_cube

#检查相位两两间，如果他们有包含相同的movement，那么关系就置为1，否则为0
    def _get_phase_competition_mask(self, phase_2_passable_lanelink):
        mask = torch.zeros((self.num_phase, self.num_phase - 1), dtype=torch.int64).to(Config.DEVICE)
        for phase_idx in range(self.num_phase):
            continuous_jdx = 0
            for phase_jdx in range(self.num_phase):
                if phase_idx == phase_jdx:
                    continue
                for lanelink_idx in range(self.num_lanelink):
                    if phase_2_passable_lanelink[phase_idx][lanelink_idx] == phase_2_passable_lanelink[phase_jdx][lanelink_idx] == 1:
                        mask[phase_idx][continuous_jdx] = 1
                continuous_jdx += 1
        return mask
