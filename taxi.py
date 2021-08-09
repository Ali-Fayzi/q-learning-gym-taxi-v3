import gym
import numpy as np
import random
from matplotlib import pyplot as plt 
env = gym.make("Taxi-v3").env

actions = env.action_space.n
states  = env.observation_space.n
#initialize Q Table 
q_table = np.zeros([states,actions])
# hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1

#Plotting Metric
reward_list = []
dropout_list = []
episode_number = 10000
for i in range(episode_number):
    #initialize envirement
    state = env.reset()
    reward_count = 0
    dropouts = 0 

    while True:
        #find action = > exploitation / exploration
        # 10% explore , 90% exploit
        if(random.uniform(0,1) < epsilon):
            action = env.action_space.sample()
        else:
            # read action from Q table 
            action = np.argmax(q_table[state])
        
        #perform action and take reward/observation
        next_state , reward , done , _ = env.step(action)

        #Q Learning Function
        old_value = q_table[state,action]
        next_max = np.max(q_table[next_state])
        next_value = (1-alpha) * old_value + alpha*(reward+gamma*next_max)
        #update Q Table  
        q_table[state,action] = next_value
        #update state
        state = next_state
        #find wrong dropout
        if reward == -10:
            dropouts +=1

        if done :
            break   
        reward_count += reward

    dropout_list.append(dropouts)
    reward_list.append(reward_count)
    if i % 100 ==0:
        print("Episode :{} , Reward : {} , Wrong Dropout:{}".format(i,reward_count,dropouts))

fig , axs = plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")


axs[1].plot(dropout_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropouts")

axs[0].grid(True)
axs[1].grid(True)

plt.show()