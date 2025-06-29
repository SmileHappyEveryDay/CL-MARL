import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiScaleEncoder(nn.Module):
    """ 多尺度状态编码器 """

    def __init__(self, args):
        super().__init__()
        # 微观编码器（处理单位级特征）
        self.micro_encoder = nn.Sequential(
            nn.Linear(args.obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # 宏观编码器（处理全局状态）
        self.macro_encoder = nn.Sequential(
            nn.Linear(args.state_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # 特征融合层
        self.fusion = nn.Linear(32 * 2, args.state_shape)  # 输出维度与原状态一致

    def forward(self, obs, state):
        """
        输入:
            obs: [batch * n_agents, obs_dim]
            state: [batch, state_dim]
        输出:
            fused_state: [batch, state_dim] (与原始状态同维度)
        """
        micro_feat = self.micro_encoder(obs)  # [batch*n_agents, 32]
        micro_feat = micro_feat.mean(dim=0, keepdim=True)  # 聚合智能体特征 [1, 32]

        macro_feat = self.macro_encoder(state)  # [batch, 32]
        fused = torch.cat([micro_feat.expand_as(macro_feat), macro_feat], dim=-1)  # [batch, 64]
        return self.fusion(fused)  # [batch, state_dim]

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 =nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )

        self.multi_scale_encoder = MultiScaleEncoder(args)

    def forward(self, q_values, states,obs=None):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        states = states.reshape(-1, self.args.state_shape)  # (episode_num * max_episode_len, state_shape)

        # 多尺度状态编码
        if obs is not None:
            obs = obs.reshape(-1, self.args.obs_shape)  # [episode_num * max_ep_len * n_agents, obs_dim]
            states = self.multi_scale_encoder(obs, states)  # 编码后维度不变

        w1 = torch.abs(self.hyper_w1(states))  # (1920, 160)
        b1 = self.hyper_b1(states)  # (1920, 32)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (1920, 5, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (1920, 1, 32)

        w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)
        return q_total
