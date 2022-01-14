# 002-算法文档说明文档
所阅读的是seungeunrho所给出的minimalRL，这部分的代码旨在用最简单的方式构建一个DQN的最基本模型，代码量少，不过麻雀虽小，五脏俱全。
使用OpenAI Gym中的传统增强学习任务之一CartPole作为练手的任务。
## 1.原理介绍
Agent根据环境的反馈基于E-greedy做出action，环境根据action给出reward并进入下一个状态s-prim。每一轮都要将每一步的信息(s,a,r,s_prim)存于replayBuffer中。DQN使用两个结构相同的神经网络：EvalNet和TargetNet，EvalNet使用最新的参数，从replay memory中随机选取mini_batch个sample训练，根据两个神经网络输出值的差值计算LossFunction来更新Q-EvalNet的参数，经过一定次数的迭代，将Q-EvalNet的参数复制给Q-TargetNet，从而完成学习过程。
## 2.总流程图
![avatar](DQN_struct.png)
## 3.分模块介绍
### 1.Qnet模块
#### a.代码
```python
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    #sample_action，相当于Q表的输出规则，E-greedy
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()

```
####b.功能描述
上述代码是agent部分Qnet的内容，是DQN用来生成、记录（state，action）的主要功能模块，相当于Q-learning中Q表的作用，在DQN中用神经网络替代这个Q表，在这个代码中，其使用最简单的全连接层（full conection）构建网络。
__init__(self)初始化了网络的全连接层fc的参数，由于CartPole游戏只有两种action，向左或者向右，所以最后的fc3的output参数是2，
forward函数用于网络的前向传播，中间层之间使用relu函数作为激活函数。
sample_action函数决定了Qnet所输出的action，其策略是DQN原论文中的E-greedy。模块首先调用forward函数对输入的一个state做一个输出out，out可以看作是Q-learning中Q表的输出，E-greedy策略会随机生成一个coin在（0,1）范围内，如果coin的值比epsilon大，则使用greedy策略，输出该state可获得的参数值最大的action，否则就随机选择一个action。

### 2.replayBuffer模块
#### a.代码
```python
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
#collection.deque : deque (maxlen=N)创建了一个固定长度的队列，当有新的队列已满时会自动移除最老的那条记录。

#放入transition
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        #每次随机从buffer里取出n个值作为mini_batch
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        #对每一个transition，返回他们的state，action，reward和s_prime以及done_mask
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)
```
####b.功能描述
replayBuffer是DQN中非常关键的一个组件，正是它的存在解决了RL和DL结合时的连续性问题和不收敛的问题。
DQN的更新和采样靠replayBuffer进行联系。连续多次采样，当采样数量每达到一次阈值，则进行一次更新。采样得到的样本扔进ReplayBuffer，更新使用ReplayBuffer中的数据。
其优点主要有两点，一、打消采样数据相关性。因为同一个episode前后step的数据是高度相关的，如果按照Q-learning的每个step后紧着进行更新，那么迭代过程会出现较大方差。
二、让数据分布变得更稳定。DQN是连续多次采样，然后进行一次更新。你将一沓(s,a,r,s)放在Buffer中后，那么就可以按照batch进行处理，按照batch投入神经网络中进行计算更新。在Q-learning中每次更新只使用一个tuple
这两点都是针对Q-learning的缺点提出来的。
这部分代码使用collection.deque来实现一个replayBuffer。 deque (maxlen=N)创建了一个固定长度的队列，当有新的队列已满时会自动移除最老的那条记录。

### 3.train模块
#### a.代码
```python
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
#### b.功能描述
用于训练Qnet，步骤如下：
1.首先从repalyBuffer中取出batch_size条记录
2.用当前的q网络根据记录中的action输出一个q_a值作为 q估计的值，并利用记录中的reward和q_target计算出的target作为 q现实的值
3.loss（q现实-q估计）反向传播更新Q网络参数

引入Q-target的原因：
在Q-learning中，用来更新的TD Target是r+γ∗Qmax，这里的MAXQ是即时查表获得。
DQN用神经网络把这个Q表取代掉了，那么假如我们使用被更新的network来产出Qmax，那这个TD Target是频繁变动，稳定性没了。出于这一点考虑，我们分出来targetNet和evalNet。让targetNet的更新频率比evalNet更低，那么我们的td target就不会频繁变动了。稳定性的问题就解决了。

### 4.main主程序接口
#### a.代码
```python
import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

def main():
    #初始化
    env = gym.make('CartPole-v1')#加载环境
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())#q_target
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:

            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))#把记忆输入replayBuffer
            s = s_prime#s进入下一个state

            score += r
            if done:
                break

        # replayBuffer<2000时，并不训练网络，只是前向传播，用e-greedy输出action
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())#每20轮更新一次q_target的参数
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()
```

#### b.功能描述
main模块整合所有模块功能，并实现完整的算法流程，其步骤如下：
1.初始化：加载gym-cartpole环境和初始化q、q_target以及replayBuffer，和一些关键参数的初始化。
2.生成memory：前2000轮，q网络并不进行训练，q网络只是单纯的利用E-greedy的策略输出action，memory会记下当前环境的state，以及接收到的q网络输出的对应的action，以及环境对action所作出的反应reward，然后将这些记录输入到repalyBuffer中作为初始的记忆经验池中的内容
3.训练网络：
2000轮之后，开始利用repalyBuffer经验池中的经验训练q网络，并更新Q网络参数。同时也会实时更新rB中的值
4.输出：将对应的轮次、分数、buffer中的记录数以及epsilon的值