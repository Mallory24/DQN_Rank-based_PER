#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import operator
import sys
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import mse


# ## Create DQN agent class

# In[2]:


class DQNAgent:
    def __init__(self, state_size, action_size, mode="Normal"): #mode: Normal DQN or Double DQN
        '''
        parameters setting for DQN
        arg: state_size: environment states 
        arg: action_size: environment action set
        arg: mode=(default=Normal|Double): Normal DQN or Double DQN
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.train_start = 400
        self.gamma = 0.99 #reward discount rate
        self.epsilon = 1.0 #exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.policy_model = self._build_model() 
        self.target_model = self._build_model() #Double DQN for stable trainning
        self.mode = mode

    def _build_model(self):
        '''
        network declaration
        '''
        model = Sequential()
        model.add(Dense(24, input_dim = self.state_size, kernel_initializer='random_uniform'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate, beta_1=0.9, beta_2=0.999)) 
        return model

    def act(self, state, explore=True):
        '''
        to explore new action, or exploit the action that lead to maximum value
        arg: state
        arg: expolre: explore or exploit mode, default set explore mode
        return: action that leads to the maximum predicted value
        '''
        # explore
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
       
        #exploit
        act_values = self.policy_model.predict(state)
        return np.argmax(act_values)

    def replay(self, samples, weights): 
        '''
        experience reply for trainning agent, loss measured by td-error
        arg: samples: batch of sampled experiences 
        arg: weights: importance sampling weights
        return: td-error to update experience's priority
        '''
        x_batch = []
        y_batch = []
        td_error = []
        for state, action, reward, next_state, done in samples:            
            #calculate the target Q values of each state
            current_q = self.policy_model.predict(state)
            if done == True:
                target_q = reward             
            else:
                if self.mode == "Double":
                    #poliy model: action selection
                    action = np.argmax(self.policy_model.predict(next_state))
                    #target model: action evaluation
                    future_reward = self.target_model.predict(next_state)[0][action]
                    target_q = reward + self.gamma * future_reward
                elif self.mode == "Normal":
                    #normal DQN
                    target_q = reward + self.gamma * np.amax(self.policy_model.predict(next_state))
            
            #calculate td_error for updating priority
            td_error.append(abs(current_q[0][action] - target_q))
            
            #replace current_q with target_q 
            current_q[0][action] = target_q 
            
            x_batch.append(state[0])
            y_batch.append(current_q[0])
        
        #train model to get closer to target q values
        self.policy_model.fit(np.array(x_batch), np.array(y_batch), epochs = 1, sample_weight = weights, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return td_error

 
        
        


# ## Binary Heap Structure

# In[3]:


class BinaryHeap(object):
    def __init__(self, buffer_size, replace=True):
        '''
        priority queue as a binary heap structure
        arg: buffer_size:  memory buffer size
        arg: replace: default True
        '''
        self.max_size = buffer_size
        self.queue_size = 0
        self.priority_queue = {}  #{1: (td-error, e_id),.. }
        self.rank_to_experience = {}
        self.replace = replace
    
    def isFull(self):
        if self.queue_size > self.max_size:
            return True
        
    def get_max_priority(self):
        '''
        new experience always gets prioritized to ensure unseen event is replayed
        return: 1
        '''
        if self.queue_size == 0: #no entry before
            return 1
        else:
            return self.priority_queue[1][0] #always get the top priority
        
    def add(self, priority, e_id):
        '''
        add new experience to priority queue
        arg: priority: td-error as priority
        arg: e_id: experience id
        return: True
        '''
        self.queue_size += 1
        if self.isFull() and self.replace == False:
            sys.exit('Error: priority queue is full and replace is set to FALSE!\n')
            return False
        self.tmp_rank = min(self.queue_size, self.max_size)
        
        self.priority_queue[self.tmp_rank] = (priority, e_id)
        self.rank_to_experience[self.tmp_rank] = e_id
        self.up_heap(self.tmp_rank)
        return True
        
    def update(self, priority, e_id): 
        '''
        update old experience with its new priority, or do insert for new experience
        arg: priority: td-error as priority
        arg: e_id: experience id
        '''
        if e_id in self.rank_to_experience.values(): #old experience, do update
            inv = {v: k for k, v in self.rank_to_experience.items()}
            rank_id = inv[e_id]
            self.priority_queue[rank_id] = (priority, e_id)
            self.rank_to_experience[rank_id] = e_id
            
            self.down_heap(rank_id)
            self.up_heap(rank_id)
            return True     
        else: #new experience, do insert
            return self.add(priority, e_id)
            
    
    def up_heap(self, node):
        '''
        upheap balance
        arg: node: current rank node
        '''
        if node > 1:
            parent = node // 2
            if self.priority_queue[node][0] >= self.priority_queue[parent][0]:
                tmp = self.priority_queue[parent]
                self.priority_queue[parent] = self.priority_queue[node]
                self.priority_queue[node] = tmp
                #change rank_to_experience
                self.rank_to_experience[parent] = self.priority_queue[parent][1]
                self.rank_to_experience[node] = self.priority_queue[node][1]
                self.up_heap(parent)
        
    def down_heap(self, node):
        '''
        downheap balance
        arg: node: current rank node
        '''
        if node < self.queue_size:
            biggest = node
            left = node * 2
            right = node * 2 + 1
            if left < self.queue_size and self.priority_queue[node][0] < self.priority_queue[left][0]:
                biggest = left
            if right < self.queue_size and self.priority_queue[node][0] < self.priority_queue[right][0]:
                biggest = right
        
            if biggest != node:
                tmp = self.priority_queue[biggest]
                self.priority_queue[biggest] = self.priority_queue[node]
                self.priority_queue[node] = tmp
                #change rank_to_experience
                self.rank_to_experience[biggest] = self.priority_queue[biggest][1]
                self.rank_to_experience[node] = self.priority_queue[node][1]
                self.down_heap(biggest)

    def rebalance(self, full=False):
        '''
        sort binary heap 
        '''
        if full:
            sorted_list = sorted(self.priority_queue.values(), key=lambda x: x[0], reverse=True)
            self.priority_queue.clear()
            self.rank_to_experience.clear()
            rank = 1
            while rank <= self.queue_size:
                self.priority_queue[rank] = sorted_list[rank-1]
                self.rank_to_experience[rank] = sorted_list[rank-1][1]
                rank += 1
            for i in range(int(self.queue_size // 2), 1, -1):
                self.down_heap(i)
    
    def get_experience_id(self, rank_ids):
        '''
        retrieve experience ids 
        arg: priority_ids: list of rank ids
        return: experience ids
        '''
        return [self.rank_to_experience[i] for i in rank_ids]
    


# ## Rank Proritized Replay

# In[4]:


class RankBuffer:
    '''
    memory buffer for rank-based prioritized experience replay 
    '''
    def __init__(self, minibatch):
        self.total_steps = 100000
        self.train_start= 400    
        self.memory = {}
        self.memory_size = 20000
        self.record_size = 0
        self.index = 0
        self.priority_queue = BinaryHeap(self.memory_size)
        self.isFull = False
        self.replace = True
        
        self.alpha = 0.7 #alpha: 1 to 0 (how much a transition is to be reused)
        self.beta= 0.7 #beta: 0 to 1 (how much importance to give for prioritized experience)
        self.k = minibatch
        self.n_partitions = 400 # partition number N, split total size to N part
            
        self.distributions = self.build_distributions()
        self.beta_grad = (1 - self.beta) / (self.total_steps - self.train_start)
    
    def get_index(self):
        '''
        get index for new experience in the memory buffer
        return: index 
        '''
        if self.record_size <= self.memory_size:
            self.record_size += 1
        
        if len(self.memory) == self.memory_size:
            self.isFull = True
            #when memory is full, reconstruct binary heap
            self.priority_queue.rebalance(self.isFull)
            if self.replace:
                self.index = 1
                return self.index
            else:
                sys.exit('Error: memory buffer is full and replace is set to FALSE!\n')
                return -1        
        else:
            self.index += 1
            return self.index
        
    def store(self, experience):
        '''
        store experience for replay
        arg: experience: <s, a, r, s', done>
        '''
        insert_index = self.get_index()
        if insert_index > 0:
            if insert_index in self.memory:
                del self.memory[insert_index]
            self.memory[insert_index] = experience
            # add to priority queue
            priority = self.priority_queue.get_max_priority()
            self.priority_queue.update(priority, insert_index)
        else:
            sys.exit('Error: store failed!\n')
            return False

    
    def check_cross(self, segment): 
        '''
        check which segment contains rank-id of previous segment;
        rank-id has a probability big enough to include up to "cross" segment but not fully cover
        arg: segment: segments as numpy array
        return: cross: the cross segement as numpy array
        '''
        segment = list(segment)
        cross = []
        for i in range(len(segment)):
            if i == (len(segment)-1):
                break
            elif segment[i] != segment[i+1]:
                cross.append(int(segment[i] + 1))
        cross = np.asarray(cross)
        return cross

    def build_distributions(self):
        '''
        pre-compute probability distribution of rank-based PER 
        '''
        distributions = {}
        partition_id = 1
        partition_size = int (math.floor(self.memory_size / self.n_partitions))    
        
        for n in range(partition_size, self.memory_size + 1, partition_size):
            distribution = {}         
            # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
            pmf = np.array([(1/i)**self.alpha for i in range(1, n + 1)]) 
            pmf_sum = pmf.sum()
            pmf_norm = list(map(lambda x: x / pmf_sum, pmf)) 
            distribution['pmf'] = pmf_norm  
            cdf = np.cumsum(pmf_norm)
            boundary = cdf[-1]/self.k  
            segment = cdf//boundary # which of the segment does each P(i) fall into
            cross_segment = self.check_cross(segment)

            ranges = [] # list of tuples (left,right) that are inclusive ranges of rank
            prev_range = (1,1) 
            for i in range(self.k):
                seg_point = np.nonzero(segment == (i+1))[0]
                cross_point = np.nonzero(cross_segment == (i+1))[0]
                if len(seg_point) > 0:
                    if len(cross_point) > 0: #if the segment point is also where the cross point locates at
                        this_range = (prev_range[-1], seg_point[-1]+1) #include the last object from previous range to this range 
                        ranges.append(this_range)
                        prev_range = this_range  
                    else:
                        this_range = (seg_point[0]+1,seg_point[-1]+1)

                        ranges.append(this_range)               
                else: 
                    if len(cross_point) > 0: #if it is only a cross point, range is from previous one to the next one
                        this_range = (prev_range[-1], prev_range[-1]+1)
                        ranges.append(this_range)
                        prev_range = (prev_range[-1]+1, prev_range[-1]+1) #replace the previous range with the next range index
                    else:
                        ranges.append(prev_range)
            distribution["ranges"] = ranges
            distributions[partition_id] = distribution 
            partition_id += 1 
        return distributions
      
               
    def retrieve(self, e_id):
        '''
        get experience from memory by indexing experience id
        return: list of memory tuples
        '''
        return [self.memory[i] for i in e_id]
    
    def sample(self, global_step): 
        '''
        sample experience for prioritized memory reply
        arg: global_step: current time step
        return: experience id, experience, importance weights
        '''
        dist_index = int (math.floor(self.record_size / self.memory_size * self.n_partitions))
        partition_size = int (math.floor(self.memory_size / self.n_partitions))
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]
        rank_ids = []     
        
        for x in distribution["ranges"]:
            sampled_object = random.sample(range(x[0],x[-1]+1), 1) 
            rank_ids.append(sampled_object[0])
        
        # beta, increase by global_step, max 1
        beta = min(self.beta + (global_step - self.train_start - 1) * self.beta_grad, 1)
        alpha_pow = [distribution['pmf'][v - 1] for v in rank_ids]
        # w = (P(i) * N) ^ (-beta) / max w
        weights = np.power(np.array(alpha_pow) * partition_max, -beta)
        weights = np.divide(weights, max(weights))
        
        e_id = self.priority_queue.get_experience_id(rank_ids)
        experience = self.retrieve(e_id)
        return (e_id, experience, weights)


    def update_priority(self, e_id, td_error):
        '''
        update sampled experience priority with new td-error
        arg: e_id: experience id
        arg: td_error: corresponding td-error
        '''
        for i in range(0, len(e_id)):
            self.priority_queue.update(abs(td_error[i]), e_id[i])


# ## Training / Testing the DQN agent

# In[ ]:


if __name__ == "__main__":
    #initialize environment and the agent
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    rank = RankBuffer(batch_size)
    #pre-compute distributions  
    distributions=rank.build_distributions()
   
    episode_test = []
    score_test = []

    
    #iterate through the game episode
    for e in range(1000):
        #reset state 
        state = env.reset()
        state = np.reshape(state,[1,-1])    
        for time_step in range(300):
            action = agent.act(state)
            #gather information from environment after taking current action
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1,-1])
            reward_true = reward / 200
            #put experience into the replay memory
            experience = (state, action, reward_true, next_state, done)
            rank.store(experience)
                          
            #make next state the current state
            state = next_state
            
            if done:
                break

            if len(rank.memory) > agent.train_start:
                e_id, experience, weights = rank.sample(time_step)
                #print('sample experience:', e_id)
                td_error = agent.replay(experience, weights)
                #print('td_error for sample:', td_error)
                #p = td_error.index(max(td_error))
                #print('expereience has max td_error after update:', e_id[p])
                rank.update_priority(e_id, td_error)
                #print('priority queue after re-rank', rank.priority_queue.priority_queue)

        
        #update target model's weight to policy model's after every episode
        if agent.mode == "Double":
            agent.target_model.set_weights(agent.policy_model.get_weights())
            
    

        #Evaluation Part: every 10 epsiodes, evaluate on agent's policy            
        if e%10 == 0:
            score = 0
            episode_test.append(e)
            for eval_e in range(50): #evaluate on 50 episodes's average rewards
                state = env.reset()
                state = np.reshape(state,[1,-1])
                for eval_time_step in range(300):
                    action = agent.act(state, False) 
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1,-1])
                    state = next_state
                    if done:
                        score += eval_time_step
                        break
            avg_score = score/50
            score_test.append(avg_score)
            print("trainning episode: {}/{}, avg score of 50 evaluation episodes: {}".format(e, 1000, avg_score))

