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

#训练部分
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        #取出32个sample的信息

        q_out = q(s)#输出qnet网络对state的对应的输出q_out
        q_a = q_out.gather(1,a)#按照memory中的action值选取对应的q_out值作为q_a作为最终输出
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)#q_target对下一个state进行预测的值里的最大值
        target = r + gamma * max_q_prime * done_mask#Q现实的值，经验值的r和q_target预测值的结合，q_target每20轮更新一次

        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()#网络参数更新

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