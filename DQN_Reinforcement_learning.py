import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyper parameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9  # greedy policy  贪婪度，0.9 的概率选择最佳策略，0.1 探索
GAMMA = 0.9  # reward discount  奖励递减值
TARGET_REPLACE_ITER = 100  # target update frequency    Q 现实网络的更新频率
MEMORY_CAPACITY = 2000  # 记忆上限
# 导入 OpenAI_Gym 实验的模拟场所，把杆子立起来的实验
env = gym.make('CartPole-v0')
env = env.unwrapped
# 环境状态有四个：小车位置、小车速率、杆子角度、杆子角速度
N_ACTIONS = env.action_space.n  # 杆子能做的动作
N_STATES = env.observation_space.shape[0]  # 杆子能获取的环境信息数
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                              int) else env.action_space.sample().shape  # to confirm the shape


class Net(nn.Module):
    # DQN 当中的神经网络模式，我们将依据这个模式建立两个神经网络，一个是现实网络 (Target Net)，一个是估计网络 (Eval Net)
    def __init__(self):
        super(Net, self).__init__()
        # 接受观测值
        self.fc1 = nn.Linear(N_STATES, 10)
        # 输入层随机生成初始值，符合(0, 0.1)的正态分布
        self.fc1.weight.detach().normal_(0, 0.1)  # 正态分布生成初始值
        # 输出动作的价值
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.detach().normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        # 初始化两个网络，一个 target 一个 evaluate
        self.eval_net, self.target_net = Net(), Net()
        # 其他的一些属性
        self.learn_step_counter = 0  # for target updating  学习到多少步
        self.memory_counter = 0  # for storing memory
        # memory 记忆库采取覆盖更新，每次覆盖最老的数据，用 counter 定位最老的数据
        # 初始化记忆库，全 0    存储的东西在store_transition可见，库大小为两个状态加上动作和奖励的行数，能存这么多批记忆
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # x 是观测值
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        # 随机选取动作的概率，用 EPSILON，如果这个概率小于随机数，就采取贪婪
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            # 选取里面最大的价值
            action = torch.max(actions_value, 1)[1].detach().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random 随机选取
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        # 记忆库存储的东西，状态、动作、奖励、下一个状态   hstack 按列顺序堆叠数组
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory    超过上限就覆盖老记忆
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 学习存储好的记忆
        # target parameter update   Q 现实的网络要隔多少步更新一下
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # 将 eval_net 复制到 target_net 当中
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # target_net 是按频率更新，eval_net 每一步学习都在更新
        # sample batch transitions
        # 从记忆库中随机抽取 BATCH_SIZE 个记忆
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # 将 memory 里面每个属性分出来方便后续进行学习
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # 输入当前动作的状态，生成所有 value，选择当初施加动作的那个价值
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # q_next 即下一步的 q 等于下一步的状态输入到 target_net 中
        # q_next 不进行反向传递误差, 所以 detach
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        # target 获得 q 现实，等于 q_next * 奖励递减值 + 当初获得的奖励，但现在获得的 q_next 是经过了神经网络的，
        # 他是对于每个动作的 q_next，要选择最大的动作，max 函数返回两个结果，[0] 是最大值，[1] 是索引
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        # 修改之后的 reward，杆子越偏旁边车越偏旁边 reward 越小
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
